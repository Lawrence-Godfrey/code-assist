"""
Core data structures for managing embeddings and search results.

This module defines the EmbeddingUnit class for storing and manipulating
embedding vectors along with their metadata, and the SearchResult class
for representing search matches with similarity scores.
"""

from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar, Union

import numpy as np

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
