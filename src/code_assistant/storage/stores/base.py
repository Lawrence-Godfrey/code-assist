"""
Base storage interfaces for both code and document storage systems.

This module provides the fundamental storage interfaces that both code and
document storage systems will implement, ensuring consistent patterns while
maintaining separation of concerns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, TypeVar, Union

import numpy as np
from pymongo.collection import Collection
from pymongo.results import DeleteResult
from pymongo.synchronous.command_cursor import CommandCursor
from pymongo.synchronous.cursor import Cursor

from code_assistant.models.embedding import EmbeddingModel


T = TypeVar("T")


@dataclass
class EmbeddingUnit:
    """
    Represents an embedding vector along with metadata about how it was generated.
    This allows tracking which model was used and with what parameters.
    """

    vector: np.ndarray
    model_name: str

    def __init__(self, vector: Union[List[float], np.ndarray], model_name: str):
        """
        Initialize an EmbeddingUnit from either a list or numpy array.

        Args:
            vector: The embedding vector as a list or numpy array
            model_name: The name of the embedding model used
        """
        if isinstance(vector, list):
            self.vector = np.array(vector)
        elif isinstance(vector, np.ndarray):
            self.vector = vector
        else:
            raise ValueError("Invalid type for embedding vector")

        self.model_name = model_name

    def __list__(self) -> List[float]:
        """Convert the embedding vector to a list."""
        return self.vector.tolist()

    def __len__(self) -> int:
        """Get the length of the embedding vector."""
        return len(self.vector)

    def __eq__(self, other):
        if not isinstance(other, EmbeddingUnit):
            return False
        return (
            np.array_equal(self.vector, other.vector)
            and self.model_name == other.model_name
        )

    def to_dict(self) -> dict:
        """Convert the EmbeddingUnit to a dictionary."""
        return {"vector": self.vector.tolist(), "model_name": self.model_name}

    @classmethod
    def from_dict(cls, data: dict) -> Optional["EmbeddingUnit"]:
        """Create an EmbeddingUnit from a dictionary."""
        if data is None:
            return None
        return cls(vector=np.array(data["vector"]), model_name=data["model_name"])


@dataclass
class SearchResult(Generic[T]):
    """
    Generic container for search results that includes both the item and its
    similarity score. Works with both code units and documents.
    """
    item: T
    similarity_score: float


# TODO: Fix ordering of methods here
class StorageBase(ABC, Generic[T]):
    """
    Abstract base class for all storage backends (e.g. code or document).
    Defines the common interface that all storage implementations must follow.
    """

    def __init__(self, namespace: str):
        """
        Initialize the storage system.

        Args:
            namespace: Identifier for the storage space (e.g., codebase name or
                     confluence space key)
        """
        self.namespace = namespace

    @abstractmethod
    def namespace_exists(self) -> bool:
        """Check if the namespace has already been initialized."""
        pass

    @abstractmethod
    def delete_namespace(self) -> None:
        """Delete all items associated with this namespace."""
        pass

    @abstractmethod
    def save_item(self, item: T) -> None:
        """
        Save a single item to storage.

        Args:
            item: The item to save
        """
        pass

    @abstractmethod
    def get_item_by_id(self, item_id: str) -> Optional[T]:
        """
        Retrieve an item by its ID.

        Args:
            item_id: The unique identifier of the item

        Returns:
            The item if found, None otherwise
        """
        pass

    @abstractmethod
    def get_items_by_type(self, item_type: str) -> List[T]:
        """
        Get all items of a specific type.

        Args:
            item_type: The type of items to retrieve

        Returns:
            List of items matching the specified type
        """
        pass

    @abstractmethod
    def refresh_vector_indexes(self, force_recreate: bool = False) -> None:
        """
        Recreate vector search indexes for all embedding models.

        Args:
            force_recreate: If True, recreate existing indexes
        """
        pass

    @abstractmethod
    def vector_search(
        self,
        embedding: EmbeddingUnit,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchResult[T]]:
        """
        Find similar items using vector similarity search.

        Args:
            embedding: The query embedding
            embedding_model: The model used to generate embeddings
            top_k: Maximum number of results to return
            threshold: Optional similarity threshold (0-1)

        Returns:
            List of search results with similarity scores
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get the total number of items in storage."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Iterate through all items in storage."""
        pass

    @abstractmethod
    def _setup_indexes(self) -> None:
        """Setup database collections/tables with proper indexes."""
        pass

    @abstractmethod
    def _ensure_vector_index(
        self,
        model_name: str,
        dimensions: int,
        force_recreate: bool
    ) -> None:
        """
        Ensure vector search index exists for a specific model.

        Args:
            model_name: Name of the embedding model
            dimensions: Number of dimensions in the embedding vector
            force_recreate: If True, recreate the index even if it exists
        """
        pass


class FilteredCollection(ABC):
    """Abstract base class for filtered MongoDB collections."""

    def __init__(self, collection: Collection, namespace: Optional[str] = None):
        """
        Initialize the filtered collection.

        Args:
            collection: The underlying MongoDB collection
            namespace: The namespace to filter by (if None, no filtering is applied)
        """
        self._collection = collection
        self._namespace = namespace

    @abstractmethod
    def _add_namespace_filter(self, filter_dict: Optional[Dict] = None) -> Dict:
        """Add namespace filter to the query."""
        pass

    @abstractmethod
    def _get_projection_fields(self) -> Dict[str, int]:
        """Get the fields to include in projections for this collection type."""
        pass

    def _get_projection(
        self, projection: Optional[Dict] = None, include_db_id=False
    ) -> Optional[Dict]:
        """
        Create projection dict that excludes _id by default unless include_db_id is True.

        Args:
            projection: Optional projection dict to merge with _id setting
            include_db_id: Whether to include MongoDB's _id field

        Returns:
            Updated projection dict or None if no projection needed
        """
        if projection is None and not include_db_id:
            return {"_id": 0}
        elif not include_db_id:
            projection = dict(projection)
            projection["_id"] = 0
        return projection

    def find(
        self, filter_dict: Optional[Dict] = None, include_db_id=False, *args, **kwargs
    ) -> Cursor:
        """Override find to include namespace filter."""
        return self._collection.find(
            self._add_namespace_filter(filter_dict),
            self._get_projection(kwargs.get("projection"), include_db_id),
            *args,
            **kwargs,
        )

    def find_one(
        self, filter_dict: Optional[Dict] = None, include_db_id=False, *args, **kwargs
    ) -> Optional[Dict]:
        """Override find_one to include namespace filter."""
        return self._collection.find_one(
            self._add_namespace_filter(filter_dict),
            self._get_projection(kwargs.get("projection"), include_db_id),
            *args,
            **kwargs,
        )

    def count_documents(
        self, filter_dict: Optional[Dict] = None, *args, **kwargs
    ) -> int:
        """Override count_documents to include namespace filter."""
        return self._collection.count_documents(
            self._add_namespace_filter(filter_dict), *args, **kwargs
        )

    def delete_many(
        self, filter_dict: Optional[Dict] = None, *args, **kwargs
    ) -> DeleteResult:
        """Override delete_many to include namespace filter."""
        return self._collection.delete_many(
            self._add_namespace_filter(filter_dict), *args, **kwargs
        )

    @abstractmethod
    def _prepare_document(self, document: Dict) -> Dict:
        """Prepare a document for insertion/replacement by adding namespace."""
        pass

    def replace_one(
        self, filter_dict: Dict, replacement: Dict, *args, **kwargs
    ) -> Any:
        """Override replace_one to include namespace filter and set namespace in document."""
        filter_dict = self._add_namespace_filter(filter_dict)
        replacement = self._prepare_document(replacement)
        return self._collection.replace_one(filter_dict, replacement, *args, **kwargs)

    def insert_one(self, document: Dict, *args, **kwargs) -> Any:
        """Override insert_one to set namespace in document."""
        document = self._prepare_document(document)
        return self._collection.insert_one(document, *args, **kwargs)

    def aggregate(
        self, pipeline: List[Dict], include_db_id=False, *args, **kwargs
    ) -> CommandCursor[Mapping[str, Any] | Any]:
        """Override aggregate to include namespace filter in $match stage."""
        # For vector search, we'll handle the namespace filter differently
        is_vector_search = pipeline and "$vectorSearch" in pipeline[0]

        if self._namespace is not None and not is_vector_search:
            # Regular aggregation pipeline handling
            if not pipeline or "$match" not in pipeline[0]:
                pipeline.insert(0, {"$match": self._add_namespace_filter()})
            elif "$match" in pipeline[0]:
                pipeline[0]["$match"].update(self._add_namespace_filter())

        # Handle projection
        projection = kwargs.pop("projection", {})
        projection.update(self._get_projection_fields())
        proj_dict = self._get_projection(projection, include_db_id)

        # Only include vectorSearchScore if we're doing a vector search
        if is_vector_search:
            proj_dict["score"] = {"$meta": "vectorSearchScore"}

        # Add projection as final stage of pipeline
        pipeline.append({"$project": proj_dict})

        return self._collection.aggregate(pipeline, *args, **kwargs)

    # Pass-through methods that don't need namespace filtering
    def create_index(self, *args, **kwargs) -> str:
        return self._collection.create_index(*args, **kwargs)

    def create_search_index(self, *args, **kwargs) -> str:
        return self._collection.create_search_index(*args, **kwargs)

    def list_search_indexes(self, *args, **kwargs) -> CommandCursor[Mapping[str, Any]]:
        return self._collection.list_search_indexes(*args, **kwargs)

    def drop_search_index(self, index_name: str, *args, **kwargs) -> None:
        return self._collection.drop_search_index(index_name, *args, **kwargs)

    @property
    def name(self) -> str:
        """Get the name of the underlying collection."""
        return self._collection.name