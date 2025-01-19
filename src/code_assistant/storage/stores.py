import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel
from pymongo.results import DeleteResult
from pymongo.synchronous.command_cursor import CommandCursor
from pymongo.synchronous.cursor import Cursor

from code_assistant.embedding.models.models import EmbeddingModel
from code_assistant.logging.logger import get_logger
from code_assistant.storage.codebase import (
    Class,
    CodeEmbedding,
    CodeUnit,
    File,
    Function,
    Method,
)

logger = get_logger(__name__)


class CodeStore(ABC):
    """Abstract base class for different storage backends."""

    def __init__(self, codebase: str):
        self.codebase = codebase

    @abstractmethod
    def codebase_exists(self) -> bool:
        """Check if a codebase has already been loaded."""
        pass

    @abstractmethod
    def delete_codebase(self) -> None:
        """Delete all files associated with a codebase."""
        pass

    @abstractmethod
    def save_unit(self, unit: CodeUnit) -> None:
        """Save a single code unit."""
        pass

    @abstractmethod
    def get_unit_by_id(self, unit_id: str) -> Optional[CodeUnit]:
        """Retrieve a code unit by its ID."""
        pass

    @abstractmethod
    def get_units_by_type(self, unit_type: Union[str, List[str]]) -> List[CodeUnit]:
        """Get all code units of specified type(s)."""
        pass

    @abstractmethod
    def iter_files(self) -> Iterator[File]:
        """Iterate through all files."""
        pass

    @abstractmethod
    def iter_classes(self) -> Iterator[Class]:
        """Iterate through all classes."""
        pass

    @abstractmethod
    def iter_methods(self) -> Iterator[Method]:
        """Iterate through all methods."""
        pass

    @abstractmethod
    def iter_functions(self) -> Iterator[Function]:
        """Iterate through all standalone functions."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the total number of code units across all files.
        This allows using len(codebase_snapshot).
        """
        pass

    def __iter__(self) -> Iterator[CodeUnit]:
        """Iterates through all code units."""
        return self.iter_flat()

    def iter_flat(self) -> Iterator[CodeUnit]:
        """Iterate through all units."""
        yield from self.iter_files()
        yield from self.iter_classes()
        yield from self.iter_methods()
        yield from self.iter_functions()

    @abstractmethod
    def vector_search(
        self,
        embedding_model: EmbeddingModel,
        embedding: CodeEmbedding,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[CodeUnit]:
        """
        Find similar code units using vector similarity search.
        Not all storage backends will support this natively.
        """
        pass


class JSONCodeStore(CodeStore):
    """Implementation of Storage using JSON files."""

    def __init__(self, codebase: str, filepath: Path):

        # Must be a json file
        super().__init__(codebase)

        if filepath.suffix != ".json":
            raise ValueError(f"Invalid file type: {filepath}. Must be a JSON file.")

        # Validate file path
        if not filepath.exists():
            # Create file
            with open(filepath, "w") as f:
                json.dump([], f)

        if not filepath.is_file():
            raise ValueError(f"Invalid file path: {filepath}. Must be a file.")

        self.filepath = filepath
        self._cache = None

    def _load_cache(self) -> None:
        """Load the entire JSON file into memory if not already cached."""
        if self._cache is None:
            with open(self.filepath, "r") as f:
                data = json.load(f)
                # Convert JSON data back into CodeUnit objects
                self._cache = [File.from_dict(file_data) for file_data in data]

    def save_unit(self, unit: CodeUnit) -> None:
        """Save by rewriting the entire JSON file."""
        self._load_cache()  # Ensure cache is loaded

        if isinstance(unit, File):
            # For files, either update existing or append new
            for i, cached_file in enumerate(self._cache):
                if cached_file.id == unit.id:
                    self._cache[i] = unit
                    break
            else:  # File not found, append it
                self._cache.append(unit)
        else:
            # For non-file units, find their parent file and update appropriately
            found = False
            for file in self._cache:
                for existing_unit in file.iter_flat():
                    if existing_unit.id == unit.id:
                        # Update the unit within its parent structure
                        if isinstance(unit, Method):
                            unit.class_ref.add_method(unit)
                        elif isinstance(unit, (Function, Class)):
                            file.add_code_unit(unit)
                        found = True
                        break
                if found:
                    break

            if not found:
                raise ValueError(
                    f"Cannot save {unit.unit_type} unit without a parent file. "
                    "Add the parent file first."
                )

        # Save entire cache back to file
        with open(self.filepath, "w") as f:
            json.dump([file.to_dict() for file in self._cache], f, indent=2)

    def get_unit_by_id(self, unit_id: str) -> Optional[CodeUnit]:
        self._load_cache()
        for file in self._cache:
            for unit in file.iter_flat():
                if unit.id == unit_id:
                    return unit
        return None

    def get_units_by_type(self, unit_type: Union[str, List[str]]) -> List[CodeUnit]:
        self._load_cache()
        if isinstance(unit_type, str):
            unit_type = [unit_type]

        results = []
        for file in self._cache:
            for unit in file.iter_flat():
                if unit.unit_type in unit_type:
                    results.append(unit)
        return results

    def codebase_exists(self) -> bool:
        """Check if a codebase exists in the JSON store."""
        self._load_cache()
        return any(file.codebase == self.codebase for file in self._cache)

    def delete_codebase(self) -> None:
        """Delete all files from a specific codebase."""
        self._load_cache()
        self._cache = [file for file in self._cache if file.codebase != self.codebase]
        # Save updated cache
        with open(self.filepath, "w") as f:
            json.dump([file.to_dict() for file in self._cache], f, indent=2)

    def iter_files(self) -> Iterator[File]:
        self._load_cache()
        yield from self._cache

    def iter_classes(self) -> Iterator[Class]:
        self._load_cache()
        for file in self._cache:
            yield from file.get_classes()

    def iter_methods(self) -> Iterator[Method]:
        self._load_cache()
        for file in self._cache:
            for cls in file.get_classes():
                yield from cls.methods

    def iter_functions(self) -> Iterator[Function]:
        self._load_cache()
        for file in self._cache:
            yield from file.get_functions()

    def __len__(self) -> int:
        """Returns the total number of code units across all files."""
        self._load_cache()
        return sum(len(file) for file in self._cache)

    def vector_search(
        self,
        embedding_model: EmbeddingModel,
        embedding: CodeEmbedding,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[CodeUnit]:
        """Basic vector search implementation using in-memory computation."""
        # TODO
        # from code_assistant.embedding.compare_embeddings import (
        #     EmbeddingSimilaritySearch,
        # )
        # from code_assistant.embedding.models.models import EmbeddingModelFactory
        #
        # self._load_cache()
        #
        # # Use existing similarity search
        # model = EmbeddingModelFactory.create(vector.model_name)
        # searcher = EmbeddingSimilaritySearch(snapshot, model)
        # results = searcher.find_similar(vector, top_k, threshold)
        # return [result.code_unit for result in results]


class DatabaseCodeStore(CodeStore):
    """Abstract base class for database storage implementations."""

    @abstractmethod
    def _setup_indexes(self) -> None:
        """Setup database collections/tables with proper indexes."""
        pass


class CodebaseFilteredCollection:
    """A wrapper around pymongo.Collection that automatically filters by codebase."""

    def __init__(self, collection: Collection, codebase: Optional[str] = None):
        """
        Initialize the filtered collection.

        Args:
            collection: The underlying MongoDB collection
            codebase: The codebase to filter by (if None, no filtering is applied)
        """
        self._collection = collection
        self._codebase = codebase

    def _add_codebase_filter(self, filter_dict: Optional[Dict] = None) -> Dict:
        """Add codebase filter to the query if codebase is set."""
        if not filter_dict:
            filter_dict = {}

        if self._codebase is not None:
            filter_dict["codebase"] = self._codebase

        return filter_dict

    def _get_projection(
        self, projection: Optional[Dict] = None, include_db_id=False
    ) -> Optional[Dict]:
        """
        Create projection dict that excludes _id by default unless include_db_id is True.

        Args:
            projection: Optional projection dict to merge with _id setting

        Returns:
            Updated projection dict or None if no projection needed
        """
        if projection is None and not include_db_id:
            return {"_id": 0}
        elif not include_db_id:
            projection = dict(projection)  # Create copy to avoid modifying original
            projection["_id"] = 0
        return projection

    def find(
        self, filter_dict: Optional[Dict] = None, include_db_id=False, *args, **kwargs
    ) -> Cursor:
        """Override find to include codebase filter."""
        return self._collection.find(
            self._add_codebase_filter(filter_dict),
            self._get_projection(kwargs.get("projection"), include_db_id),
            *args,
            **kwargs,
        )

    def find_one(
        self, filter_dict: Optional[Dict] = None, include_db_id=False, *args, **kwargs
    ) -> Optional[Dict]:
        """Override find_one to include codebase filter."""
        return self._collection.find_one(
            self._add_codebase_filter(filter_dict),
            self._get_projection(kwargs.get("projection"), include_db_id),
            *args,
            **kwargs,
        )

    def count_documents(
        self, filter_dict: Optional[Dict] = None, *args, **kwargs
    ) -> int:
        """Override count_documents to include codebase filter."""
        return self._collection.count_documents(
            self._add_codebase_filter(filter_dict), *args, **kwargs
        )

    def delete_many(
        self, filter_dict: Optional[Dict] = None, *args, **kwargs
    ) -> DeleteResult:
        """Override delete_many to include codebase filter."""
        return self._collection.delete_many(
            self._add_codebase_filter(filter_dict), *args, **kwargs
        )

    def aggregate(
        self, pipeline: List[Dict], *args, **kwargs
    ) -> CommandCursor[Mapping[str, Any] | Any]:
        """Override aggregate to include codebase filter in $match stage."""
        if self._codebase is not None:
            # Add codebase filter as first stage if pipeline doesn't start with $match
            if not pipeline or "$match" not in pipeline[0]:
                pipeline.insert(0, {"$match": {"codebase": self._codebase}})
            # Add to existing $match stage if it exists
            elif "$match" in pipeline[0]:
                pipeline[0]["$match"]["codebase"] = self._codebase

        return self._collection.aggregate(pipeline, *args, **kwargs)

    def replace_one(self, filter_dict: Dict, replacement: Dict, *args, **kwargs) -> Any:
        """Override replace_one to include codebase filter and set codebase in document."""
        if self._codebase is not None:
            filter_dict = self._add_codebase_filter(filter_dict)
            replacement["codebase"] = self._codebase

        return self._collection.replace_one(filter_dict, replacement, *args, **kwargs)

    def insert_one(self, document: Dict, *args, **kwargs) -> Any:
        """Override insert_one to set codebase in document."""
        if self._codebase is not None:
            document["codebase"] = self._codebase

        return self._collection.insert_one(document, *args, **kwargs)

    def create_search_index(self, *args, **kwargs) -> str:
        """Pass through create_search_index to the underlying collection."""
        return self._collection.create_search_index(*args, **kwargs)

    def list_search_indexes(self, *args, **kwargs) -> CommandCursor[Mapping[str, Any]]:
        """Pass through list_search_indexes to the underlying collection."""
        return self._collection.list_search_indexes(*args, **kwargs)

    def create_index(self, *args, **kwargs) -> str:
        """Pass through create_index to the underlying collection."""
        return self._collection.create_index(*args, **kwargs)

    @property
    def name(self) -> str:
        """Get the name of the underlying collection."""
        return self._collection.name


class MongoDBCodeStore(DatabaseCodeStore):
    """Implementation of Storage using MongoDB."""

    def __init__(
        self, codebase: str, connection_string: str, database: str = "code_assistant"
    ):
        super().__init__(codebase)

        self.client = MongoClient(connection_string)
        self.db = self.client[database]

        # Single collection for all code units
        self.code_units: CodebaseFilteredCollection = CodebaseFilteredCollection(
            self.db.code_units
        )

        self._setup_indexes()

    def _setup_indexes(self) -> None:
        """Setup MongoDB collection with proper indexes."""

        # Create indexes
        self.code_units.create_index("id", unique=True)
        self.code_units.create_index("unit_type")
        self.code_units.create_index("codebase")
        self.code_units.create_index([("unit_type", 1), ("codebase", 1)])

    def _ensure_vector_index(self, model_name: str, dimensions: int) -> None:
        """
        Create vector search index for a specific model if it doesn't exist.

        Args:
            model_name: Name of the embedding model
            dimensions: Number of dimensions in the embedding vector
        """
        # Check if index already exists for this model
        existing_indices = list(self.code_units.list_search_indexes())
        index_exists = any(
            index.get("name", "") == f"vector_index_{model_name.replace('/', '_')}"
            for index in existing_indices
        )

        if not index_exists:
            vector_path = f"embeddings.{model_name}.vector"
            search_index = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": vector_path,
                            "numDimensions": dimensions,
                            "similarity": "cosine",
                        }
                    ]
                },
                name=f"vector_index_{model_name.replace('/', '_')}",
                type="vectorSearch",
            )

            try:
                logger.info(f"Creating vector index for model {model_name}")
                self.code_units.create_search_index(model=search_index)
                # Wait for index to be ready
                while True:
                    indices = list(
                        self.code_units.list_search_indexes(
                            f"vector_index_{model_name.replace('/', '_')}"
                        )
                    )
                    if len(indices) and indices[0].get("queryable") is True:
                        break
                    time.sleep(0.1)
                logger.info(f"Vector index created for model {model_name}")
            except Exception as e:
                logger.warning(
                    f"Could not create vector search index for {model_name}: {str(e)}"
                )

    def save_unit(self, unit: CodeUnit) -> None:
        """
        Save or update a unit in MongoDB collection.

        If the unit already exists (by ID or name+filepath), only update that unit
        without modifying its sub-units. If it's a new unit, save it and all sub-units.

        Args:
            unit: CodeUnit to save/update (File, Class, Method, or Function)
        """
        # Check if unit exists by ID first
        existing_unit = self.code_units.find_one({"id": unit.id})

        if existing_unit:
            # Unit exists by ID - just update the unit itself
            unit_dict = unit.to_dict()
            # Remove sub-units to prevent updating them
            unit_dict.pop("code_units", None)
            unit_dict.pop("methods", None)
            self.code_units.replace_one({"id": unit.id}, unit_dict)
            return

        # If no match by ID, check for existing unit by type-specific criteria
        search_criteria = {
            "unit_type": unit.unit_type,
            "name": unit.name,
            "filepath": str(unit.filepath),
        }

        if isinstance(unit, Method):
            search_criteria["classname"] = unit.classname

        existing_unit = self.code_units.find_one(search_criteria)

        unit_dict = unit.to_dict()

        # Remove sub-units to prevent updating them
        unit_dict.pop("code_units", None)
        unit_dict.pop("methods", None)

        # Update the existing unit, or insert a new one
        self.code_units.replace_one(search_criteria, unit_dict, upsert=True)

        if not existing_unit:
            # If new unit, save all sub-units
            for sub_unit in unit:
                self.save_unit(sub_unit)

    def get_unit_by_id(self, unit_id: str) -> Optional[CodeUnit]:
        """
        Find a document by ID in MongoDB collection.
        This currently only works for files, since methods, classes and functions are within files
        """
        return CodeUnit.from_dict(self.code_units.find_one({"id": unit_id}))

    def get_units_by_type(self, unit_type: Union[str, List[str]]) -> List[CodeUnit]:
        """Get all code units of specified type(s)."""
        if isinstance(unit_type, str):
            unit_type = [unit_type]

        results = []
        for doc in self.code_units.find({"unit_type": {"$in": unit_type}}):
            if unit := CodeUnit.from_dict(doc):
                results.append(unit)
        return results

    def delete_codebase(self) -> None:
        """Delete all files associated with a codebase."""
        # Delete all documents
        self.code_units.delete_many()

        logger.info(f"Deleted codebase {self.codebase}")

    def codebase_exists(self) -> bool:
        """Check if a codebase exists in the MongoDB collection."""
        return self.code_units.count_documents({}) > 0

    def iter_files(self) -> Iterator[File]:
        """Iterate through all files."""
        for doc in self.code_units.find({"unit_type": "file"}):
            if file := CodeUnit.from_dict(doc):
                yield file

    def iter_classes(self) -> Iterator[Class]:
        """Iterate through all classes."""
        for doc in self.code_units.find({"unit_type": "class"}):
            if cls := CodeUnit.from_dict(doc):
                yield cls

    def iter_methods(self) -> Iterator[Method]:
        """Iterate through all methods."""
        for doc in self.code_units.find({"unit_type": "method"}):
            if method := CodeUnit.from_dict(doc):
                yield method

    def iter_functions(self) -> Iterator[Function]:
        """Iterate through all standalone functions."""
        for doc in self.code_units.find({"unit_type": "function"}):
            if func := CodeUnit.from_dict(doc):
                yield func

    def __len__(self) -> int:
        """Returns the total number of code units in the collection."""
        return self.code_units.count_documents({})

    def vector_search(
        self,
        embedding_model: EmbeddingModel,
        embedding: CodeEmbedding,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[CodeUnit]:
        """Perform vector similarity search across all code units."""

        # Use model-specific index
        index_name = f"vector_index_{embedding.model_name.replace('/', '_')}"
        vector_path = f"embeddings.{embedding.model_name}.vector"

        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": vector_path,
                    "queryVector": list(embedding.vector),
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                }
            }
        ]

        if threshold:
            pipeline.append({"$match": {"score": {"$gte": threshold}}})

        results = []
        for doc in self.code_units.aggregate(pipeline):
            if unit := CodeUnit.from_dict(doc):
                results.append(unit)

        return results
