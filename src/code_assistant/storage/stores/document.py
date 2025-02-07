"""
MongoDB-backed storage implementation for Confluence documents.

This module provides concrete implementations for storing and retrieving documents,
including the filtered collection wrapper and the main storage class.
"""

import time
from typing import Dict, Iterator, List, Optional

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from code_assistant.logging.logger import get_logger
from code_assistant.models.embedding import EmbeddingModel
from code_assistant.storage.document import Document
from code_assistant.storage.stores.base import FilteredCollection, StorageBase
from code_assistant.storage.types import EmbeddingUnit, SearchResult

logger = get_logger(__name__)


class DocumentFilteredCollection(FilteredCollection):
    """Collection wrapper that filters by space key."""

    def _add_namespace_filter(self, filter_dict: Optional[Dict] = None) -> Dict:
        """Add space_key filter to the query."""
        if not filter_dict:
            filter_dict = {}

        if self._namespace is not None:
            filter_dict["space_key"] = self._namespace

        return filter_dict

    def _get_projection_fields(self) -> Dict[str, int]:
        """Get the fields to include in projections for documents."""
        return {
            "id": 1,
            "space_key": 1,
            "doc_type": 1,
            "title": 1,
            "content": 1,
            "last_modified": 1,
            "metadata": 1,
            "embeddings": 1,
            "parent_id": 1,
            "version": 1,
            "url": 1,
            "page_id": 1,
            "file_type": 1,
            "size": 1,
        }

    def _prepare_document(self, document: Dict) -> Dict:
        """Add space_key to document if namespace is set."""
        if self._namespace is not None:
            document["space_key"] = self._namespace
        return document


class MongoDBDocumentStore(StorageBase[Document]):
    """MongoDB implementation for storing Confluence documents."""

    def __init__(
        self, space_key: str, connection_string: str, database: str = "code_assistant"
    ):
        """
        Initialize the MongoDB document store.

        Args:
            space_key: Confluence space key
            connection_string: MongoDB connection string
            database: Name of the database
        """
        super().__init__(namespace=space_key)

        self.client = MongoClient(connection_string)
        self.db = self.client[database]
        self.collection = DocumentFilteredCollection(self.db.documents, self.namespace)

        self._setup_indexes()

    def _setup_indexes(self) -> None:
        """Setup MongoDB collection with proper indexes."""
        self.collection.create_index("id", unique=True)
        self.collection.create_index("doc_type")
        self.collection.create_index("space_key")
        self.collection.create_index([("doc_type", 1), ("space_key", 1)])
        self.collection.create_index("last_modified")
        # Index for parent-child relationships
        self.collection.create_index([("parent_id", 1), ("space_key", 1)])
        self.collection.create_index([("page_id", 1), ("space_key", 1)])

    # TODO: Might be able to take this to base class?
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
                            "path": "space_key",
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

    def save_item(self, document: Document) -> None:
        """
        Save or update a document.

        Args:
            document: The document to save
        """
        # Check if document exists by ID
        existing_doc = self.collection.find_one({"id": document.id})

        doc_dict = document.to_dict()

        if existing_doc:
            # Document exists - update it
            self.collection.replace_one({"id": document.id}, doc_dict)
        else:
            # New document - insert it
            self.collection.insert_one(doc_dict)

    def get_item_by_id(self, item_id: str) -> Optional[Document]:
        """Find a document by ID."""
        if doc := self.collection.find_one({"id": item_id}):
            return Document.from_dict(doc)
        return None

    def get_items_by_type(self, doc_type: str) -> List[Document]:
        """Get all documents of specified type."""
        results = []
        for doc in self.collection.find({"doc_type": doc_type}):
            if document := Document.from_dict(doc):
                results.append(document)
        return results

    def namespace_exists(self) -> bool:
        """Check if space exists."""
        return self.collection.count_documents({}) > 0

    def delete_namespace(self) -> None:
        """Delete all documents in the space."""
        self.collection.delete_many({})
        logger.info(f"Deleted space {self.namespace}")

    # TODO: Might be able to take to base class
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

    # TODO: Might be able to take to base class
    def vector_search(
        self,
        embedding: EmbeddingUnit,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchResult[Document]]:
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
            if document := Document.from_dict(doc):
                results.append(SearchResult(document, similarity_score=score))

        return results

    def __len__(self) -> int:
        """Get total number of documents."""
        return self.collection.count_documents({})

    def __iter__(self) -> Iterator[Document]:
        """Iterate through all documents."""
        for doc in self.collection.find():
            if document := Document.from_dict(doc):
                yield document
