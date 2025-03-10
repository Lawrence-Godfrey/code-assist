"""
MongoDB-backed storage implementation for code units.

This module provides concrete implementations for storing and retrieving code units,
including the filtered collection wrapper and the main storage class.
"""

import time
from typing import Dict, Iterator, List, Optional, Union

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from code_assistant.logging.logger import get_logger
from code_assistant.models.embedding import EmbeddingModel
from code_assistant.storage.codebase import Class, CodeUnit, File, Function, Method
from code_assistant.storage.stores.base import FilteredCollection, StorageBase
from code_assistant.storage.types import EmbeddingUnit, SearchResult

logger = get_logger(__name__)


class CodebaseFilteredCollection(FilteredCollection):
    """Collection wrapper that filters by codebase."""

    def _add_namespace_filter(self, filter_dict: Optional[Dict] = None) -> Dict:
        """Add codebase filter to the query."""
        if not filter_dict:
            filter_dict = {}

        if self._namespace is not None:
            filter_dict["codebase"] = self._namespace

        return filter_dict

    def _get_projection_fields(self) -> Dict[str, int]:
        """Get the fields to include in projections for code units."""
        return {
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

    def _prepare_document(self, document: Dict) -> Dict:
        """Add codebase to document if namespace is set."""
        if self._namespace is not None:
            document["codebase"] = self._namespace
        return document


class MongoDBCodeStore(StorageBase[CodeUnit]):
    """MongoDB implementation for storing code units."""

    def __init__(
        self, codebase: str, connection_string: str, database: str = "code_assistant"
    ):
        """
        Initialize the MongoDB code store.

        Args:
            codebase: Name of the codebase
            connection_string: MongoDB connection string
            database: Name of the database
        """
        super().__init__(namespace=codebase)

        self.client = MongoClient(connection_string)
        self.db = self.client[database]
        self.collection = CodebaseFilteredCollection(self.db.code_units, self.namespace)

        self._setup_indexes()

    def _setup_indexes(self) -> None:
        """Setup MongoDB collection with proper indexes."""
        self.collection.create_index("id", unique=True)
        self.collection.create_index("unit_type")
        self.collection.create_index("codebase")
        self.collection.create_index([("unit_type", 1), ("codebase", 1)])

    def _ensure_vector_index(
        self, model_name: str, dimensions: int, force_recreate: bool
    ) -> None:
        """Create vector search index for a specific model if it doesn't exist."""
        # Check if index already exists
        existing_indices = list(self.collection.list_search_indexes())
        index_exists = any(
            index.get("name", "") == f"vector_index_{model_name.replace('/', '_')}"
            for index in existing_indices
        )

        if index_exists and force_recreate:
            self.collection.drop_search_index(
                f"vector_index_{model_name.replace('/', '_')}"
            )
            index_exists = False

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
                        },
                        # Add namespace field for filtering
                        {
                            "type": "filter",
                            "path": "codebase",
                        },
                    ]
                },
                name=f"vector_index_{model_name.replace('/', '_')}",
                type="vectorSearch",
            )

            try:
                logger.info(f"Creating vector index for model {model_name}")
                self.collection.create_search_index(model=search_index)
                # Wait for index to be ready
                while True:
                    indices = list(
                        self.collection.list_search_indexes(
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

    def save_item(self, unit: CodeUnit) -> None:
        """
        Save or update a code unit.

        Args:
            unit: The code unit to save
        """
        # Check if unit exists by ID
        existing_unit = self.collection.find_one({"id": unit.id})

        # Convert to dictionary, excluding sub-units
        unit_dict = unit.to_dict()
        unit_dict.pop("code_units", None)  # For File
        unit_dict.pop("methods", None)  # For Class

        if existing_unit:
            # Unit exists - update it
            self.collection.replace_one({"id": unit.id}, unit_dict)
        else:
            # New unit - insert it and its sub-units
            self.collection.insert_one(unit_dict)

            # Save sub-units if needed
            if isinstance(unit, File):
                for sub_unit in unit.code_units:
                    self.save_item(sub_unit)
            elif isinstance(unit, Class):
                for method in unit.methods:
                    self.save_item(method)

    def get_item_by_id(self, item_id: str) -> Optional[CodeUnit]:
        """Find a code unit by ID."""
        if doc := self.collection.find_one({"id": item_id}):
            return CodeUnit.from_dict(doc)
        return None

    def get_items_by_type(self, unit_type: Union[str, List[str]]) -> List[CodeUnit]:
        """Get all code units of specified type(s)."""
        if isinstance(unit_type, str):
            unit_type = [unit_type]

        results = []
        for doc in self.collection.find({"unit_type": {"$in": unit_type}}):
            if unit := CodeUnit.from_dict(doc):
                results.append(unit)
        return results

    def namespace_exists(self) -> bool:
        """Check if codebase exists."""
        return self.collection.count_documents({}) > 0

    def delete_namespace(self) -> None:
        """Delete all code units in the codebase."""
        self.collection.delete_many({})
        logger.info(f"Deleted codebase {self.namespace}")

    def iter_files(self) -> Iterator[File]:
        """Iterate through all files."""
        for doc in self.collection.find({"unit_type": "file"}):
            if file := CodeUnit.from_dict(doc):
                yield file

    def iter_classes(self) -> Iterator[Class]:
        """Iterate through all classes."""
        for doc in self.collection.find({"unit_type": "class"}):
            if cls := CodeUnit.from_dict(doc):
                yield cls

    def iter_methods(self) -> Iterator[Method]:
        """Iterate through all methods."""
        for doc in self.collection.find({"unit_type": "method"}):
            if method := CodeUnit.from_dict(doc):
                yield method

    def iter_functions(self) -> Iterator[Function]:
        """Iterate through all standalone functions."""
        for doc in self.collection.find({"unit_type": "function"}):
            if func := CodeUnit.from_dict(doc):
                yield func

    def iter_flat(self) -> Iterator[CodeUnit]:
        """Iterate through all units."""
        yield from self.iter_files()
        yield from self.iter_classes()
        yield from self.iter_methods()
        yield from self.iter_functions()

    def refresh_vector_indexes(self, force_recreate: bool = False) -> None:
        """Recreate vector search indexes for all embedding models."""
        # Find unique model names and dimensions
        model_names = {}
        for doc in self.collection.find({"embeddings": {"$exists": True}}):
            for model_name, embedding in doc.get("embeddings", {}).items():
                model_names[model_name] = len(embedding["vector"])

        # Create/update indexes for each model
        for model_name, dimensions in model_names.items():
            self._ensure_vector_index(model_name, dimensions, force_recreate)

    def vector_search(
        self,
        embedding: EmbeddingUnit,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchResult[CodeUnit]]:
        """
        Perform vector similarity search.

        Args:
            embedding: Query embedding
            embedding_model: Model used to generate embeddings
            top_k: Maximum number of results to return
            threshold: Optional similarity threshold

        Returns:
            List of search results with similarity scores
        """
        # Use model-specific index
        index_name = f"vector_index_{embedding.model_name.replace('/', '_')}"
        vector_path = f"embeddings.{embedding.model_name}.vector"
        query_vector = embedding.vector.astype(float).tolist()

        # Build aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": vector_path,
                    "queryVector": query_vector,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
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
        for doc in self.collection.aggregate(pipeline):
            score = doc.pop("score")
            if unit := CodeUnit.from_dict(doc):
                results.append(SearchResult(unit, similarity_score=score))

        return results

    def __len__(self) -> int:
        """Get total number of code units."""
        return self.collection.count_documents({})

    def __iter__(self) -> Iterator[CodeUnit]:
        """Iterate through all code units."""
        for doc in self.collection.find():
            if unit := CodeUnit.from_dict(doc):
                yield unit
