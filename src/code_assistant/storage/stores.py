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

    def add_file(self, file: File) -> None:
        """Add a file to the codebase snapshot."""
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

        # Find and replace the unit in cache
        for i, cached_unit in enumerate(self._cache):
            if cached_unit.id == unit.id:
                self._cache[i] = unit
                break

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
        self.code_units.create_index("unit_type")  # For efficient filtering by type

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
        self.code_units.replace_one({"id": unit.id}, unit.to_dict(), upsert=True)

    def get_unit_by_id(self, unit_id: str) -> Optional[CodeUnit]:
        """Find a document by ID in MongoDB collection."""
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
