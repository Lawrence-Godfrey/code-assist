import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel
from pymongo.results import DeleteResult
from pymongo.synchronous.command_cursor import CommandCursor
from pymongo.synchronous.cursor import Cursor

from code_assistant.logging.logger import get_logger
from code_assistant.models.embedding import EmbeddingModel
from code_assistant.storage.codebase import (
    Class,
    CodeEmbedding,
    CodeUnit,
    File,
    Function,
    Method,
)

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """
    A container for search results that includes both the code unit metadata
    and its similarity score. This provides a clean, type-safe way to return
    search results with their relevance scores.
    """

    code_unit: CodeUnit
    similarity_score: float


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
    def refresh_vector_indexes(self, force_recreate=False) -> None:
        """Recreate vector search indexes for all embedding models."""
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
        embedding: CodeEmbedding,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Find similar code units using vector similarity search.
        Not all storage backends will support this natively.
        """
        pass


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

    def drop_search_index(self, index_name: str, *args, **kwargs) -> None:
        """Drop an index from the collection."""
        return self._collection.drop_search_index(index_name, *args, **kwargs)

    def aggregate(
        self, pipeline: List[Dict], include_db_id=False, *args, **kwargs
    ) -> CommandCursor[Mapping[str, Any] | Any]:
        """Override aggregate to include codebase filter in $match stage."""
        # For vector search, we'll handle the codebase filter differently
        # The calling method should include it in the $vectorSearch stage
        is_vector_search = pipeline and "$vectorSearch" in pipeline[0]

        if self._codebase is not None and not is_vector_search:
            # Regular aggregation pipeline handling
            if not pipeline or "$match" not in pipeline[0]:
                pipeline.insert(0, {"$match": {"codebase": self._codebase}})
            elif "$match" in pipeline[0]:
                pipeline[0]["$match"]["codebase"] = self._codebase

        # Handle projection
        projection = kwargs.pop("projection", {})
        projection.update(
            {
                "id": 1,
                "codebase": 1,
                "unit_type": 1,
                "docstring": 1,
                "name": 1,
                "filepath": 1,
                "classname": 1,
                "source_code": 1,
                "embeddings": 1,
            }
        )
        proj_dict = self._get_projection(projection, include_db_id)

        # Only include vectorSearchScore if we're doing a vector search
        if is_vector_search:
            proj_dict["score"] = {"$meta": "vectorSearchScore"}

        # Add projection as final stage of pipeline
        pipeline.append({"$project": proj_dict})

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
            self.db.code_units, self.codebase
        )

        self._setup_indexes()

    def _setup_indexes(self) -> None:
        """Setup MongoDB collection with proper indexes."""

        # Create indexes
        self.code_units.create_index("id", unique=True)
        self.code_units.create_index("unit_type")
        self.code_units.create_index("codebase")
        self.code_units.create_index([("unit_type", 1), ("codebase", 1)])

    def refresh_vector_indexes(self, force_recreate=False) -> None:
        """
        Recreate vector search indexes for all embedding models.

        Args:
            force_recreate: If True, recreate the indexes even if they already
                exist. This can be useful if the index needs to be updated.
        """
        # find unique model names
        model_names = dict()
        for doc in self.code_units.find({"embeddings": {"$exists": True}}):
            for model_name in doc.get("embeddings", {}):
                model_names[model_name] = len(doc["embeddings"][model_name]["vector"])

        for model_name, dimensions in model_names.items():
            self._ensure_vector_index(model_name, dimensions, force_recreate)

    def _ensure_vector_index(
        self, model_name: str, dimensions: int, force_recreate: bool
    ) -> None:
        """
        Create vector search index for a specific model if it doesn't exist.
        Include codebase field in the index for filtering.

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

        if index_exists and force_recreate:
            self.code_units.drop_search_index(
                f"vector_index_{model_name.replace('/', '_')}"
            )
            index_exists = False

        if not index_exists:
            search_index = self._build_vector_index(model_name, dimensions)

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

    def _build_vector_index(self, model_name: str, dimensions: int) -> SearchIndexModel:
        """
        Create a search index for vector similarity search.

        Args:
            model_name: Name of the embedding model
            dimensions: Number of dimensions in the embedding vector

        Returns:
            SearchIndexModel object for the vector search index
        """

        vector_path = f"embeddings.{model_name}.vector"
        search_index = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": vector_path,
                        "numDimensions": dimensions,
                        "similarity": "cosine",
                    },
                    # Add codebase field for filtering
                    {
                        "type": "filter",
                        "path": "codebase",
                    },
                ]
            },
            name=f"vector_index_{model_name.replace('/', '_')}",
            type="vectorSearch",
        )

        return search_index

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
        embedding: CodeEmbedding,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Perform vector similarity search across all code units within the codebase."""
        # Use model-specific index
        index_name = f"vector_index_{embedding.model_name.replace('/', '_')}"
        vector_path = f"embeddings.{embedding.model_name}.vector"
        query_vector = embedding.vector.astype(float).tolist()

        # Include codebase filter in the vectorSearch pre-filter
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": vector_path,
                    "queryVector": query_vector,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    "filter": {"codebase": self.codebase},  # Restrict to this codebase
                }
            }
        ]

        # Add threshold filter if specified
        if threshold:
            pipeline.append(
                {
                    "$match": {
                        "$expr": {"$gte": [{"$meta": "vectorSearchScore"}, threshold]}
                    }
                }
            )

        results = []
        for doc in self.code_units.aggregate(pipeline):
            score = doc.pop("score")
            if unit := CodeUnit.from_dict(doc):
                results.append(SearchResult(unit, similarity_score=score))

        return results
