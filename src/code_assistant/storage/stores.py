import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Union

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

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

    @abstractmethod
    def codebase_exists(self, codebase_name: str) -> bool:
        """Check if a codebase has already been loaded."""
        pass

    @abstractmethod
    def delete_codebase(self, codebase_name: str) -> None:
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

    def add_file(self, codebase: str, file: File) -> None:
        """Add a file to the codebase snapshot."""
        file.codebase = codebase
        self.save_unit(file)

    def __iter__(self) -> Iterator[File]:
        """Iterates through all files."""
        return self.iter_files()

    def iter_flat(self) -> Iterator[CodeUnit]:
        """Iterate through all units."""
        yield from self.iter_classes()
        yield from self.iter_methods()
        yield from self.iter_functions()

    @abstractmethod
    def vector_search(
        self, vector: CodeEmbedding, top_k: int = 5, threshold: Optional[float] = None
    ) -> List[CodeUnit]:
        """
        Find similar code units using vector similarity search.
        Not all storage backends will support this natively.
        """
        pass


class JSONCodeStore(CodeStore):
    """Implementation of Storage using JSON files."""

    def __init__(self, filepath: Path):

        # Must be a json file
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

    def codebase_exists(self, codebase_name: str) -> bool:
        """Check if a codebase exists in the JSON store."""
        self._load_cache()
        return any(file.codebase == codebase_name for file in self._cache)

    def delete_codebase(self, codebase_name: str) -> None:
        """Delete all files from a specific codebase."""
        self._load_cache()
        self._cache = [file for file in self._cache if file.codebase != codebase_name]
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
        self, vector: CodeEmbedding, top_k: int = 5, threshold: Optional[float] = None
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


class MongoDBCodeStore(DatabaseCodeStore):
    """Implementation of Storage using MongoDB."""

    def __init__(self, connection_string: str, database: str = "code_assistant"):
        self.client = MongoClient(connection_string)
        self.db = self.client[database]
        self._setup_indexes()

    def _setup_indexes(self) -> None:
        """Setup MongoDB collection with proper indexes."""

        # Single collection for all code units
        self.code_units: Collection = self.db.code_units

        # Create indexes
        self.code_units.create_index("id", unique=True)
        self.code_units.create_index("unit_type")
        self.code_units.create_index("codebase")
        self.code_units.create_index([("unit_type", 1), ("codebase", 1)])
        self.code_units.create_index("parent_file")
        self.code_units.create_index("parent_class")

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

    def _convert_to_unit(self, data: dict) -> Optional[CodeUnit]:
        """Convert MongoDB document back to appropriate CodeUnit type."""
        if not data:
            return None

        return CodeUnit.from_dict(data)

    def save_unit(self, unit: CodeUnit) -> None:
        """Save a unit to MongoDB collection."""

        if isinstance(unit, File):
            unit_dict = unit.to_dict()
            unit_dict.pop("code_units", None)  # Remove nested code units
            db_file = self.code_units.replace_one(
                {"id": unit.id}, unit_dict, upsert=True
            )

            for code_unit in unit.code_units:
                code_unit_dict = code_unit.to_dict()
                code_unit_dict["parent_file"] = db_file.upserted_id
                code_unit_dict.pop("methods", None)

                db_code_unit = self.code_units.replace_one(
                    {"id": code_unit.id}, code_unit_dict, upsert=True
                )

                if isinstance(code_unit, Class):
                    for method in code_unit.methods:
                        method_dict = method.to_dict()
                        method_dict["parent_class"] = db_code_unit.upserted_id
                        self.code_units.replace_one(
                            {"id": method.id}, method_dict, upsert=True
                        )

    def get_unit_by_id(self, unit_id: str) -> Optional[CodeUnit]:
        """
        Find a document by ID in MongoDB collection.
        This currently only works for files, since methods, classes and functions are within files
        """
        return self._convert_to_unit(self.code_units.find_one({"id": unit_id}))

    def get_units_by_type(self, unit_type: Union[str, List[str]]) -> List[CodeUnit]:
        """Get all code units of specified type(s)."""
        if isinstance(unit_type, str):
            unit_type = [unit_type]

        results = []
        for doc in self.code_units.find({"unit_type": {"$in": unit_type}}):
            if unit := self._convert_to_unit(doc):
                results.append(unit)
        return results

    def delete_codebase(self, codebase_name: str) -> None:
        """Delete all files associated with a codebase."""
        # First find all documents with this codebase name
        files = list(self.code_units.find({"codebase": codebase_name}))
        print(files)
        # Then find all nested documents (methods, classes) and delete them
        classes = list(
            self.code_units.find(
                {
                    "parent_file": {"$in": [file["_id"] for file in files]},
                    "unit_type": "class",
                }
            )
        )
        functions = list(
            self.code_units.find(
                {
                    "parent_file": {"$in": [file["_id"] for file in files]},
                    "unit_type": "function",
                }
            )
        )
        print(f"functions: {len(list(functions))}")
        self.code_units.delete_many(
            {"_id": {"$in": [func["_id"] for func in functions]}}
        )
        print(f"file ids: {[file['_id'] for file in files]}")
        print(f"file ids: {[file['_id'] for file in files]}")
        print(f"Classes: {len(list(classes))}")
        methods = list(
            self.code_units.find(
                {"parent_class": {"$in": [cls["_id"] for cls in classes]}}
            )
        )
        print(f"Methods: {len(list(methods))}")
        # Delete all nested documents first
        self.code_units.delete_many(
            {"_id": {"$in": [method["_id"] for method in methods]}}
        )
        self.code_units.delete_many({"_id": {"$in": [cls["_id"] for cls in classes]}})
        # Delete the files
        self.code_units.delete_many({"_id": {"$in": [file["_id"] for file in files]}})
        logger.info(f"Deleted codebase {codebase_name}")

    def codebase_exists(self, codebase_name: str) -> bool:
        """Check if a codebase has already been loaded."""
        return self.code_units.count_documents({"codebase": codebase_name}) > 0

    def iter_files(self) -> Iterator[File]:
        """Iterate through all files."""
        for doc in self.code_units.find({"unit_type": "file"}):
            if file := self._convert_to_unit(doc):
                yield file

    def iter_classes(self) -> Iterator[Class]:
        """Iterate through all classes."""
        for doc in self.code_units.find({"unit_type": "class"}):
            if cls := self._convert_to_unit(doc):
                yield cls

    def iter_methods(self) -> Iterator[Method]:
        """Iterate through all methods."""
        for doc in self.code_units.find({"unit_type": "method"}):
            if method := self._convert_to_unit(doc):
                yield method

    def iter_functions(self) -> Iterator[Function]:
        """Iterate through all standalone functions."""
        for doc in self.code_units.find({"unit_type": "function"}):
            if func := self._convert_to_unit(doc):
                yield func

    def __len__(self) -> int:
        """Returns the total number of code units in the collection."""
        return self.code_units.count_documents({})

    def vector_search(
        self,
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
            if unit := self._convert_to_unit(doc):
                results.append(unit)

        return results
