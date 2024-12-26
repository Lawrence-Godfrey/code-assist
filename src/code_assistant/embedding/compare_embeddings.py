from dataclasses import dataclass

import numpy as np
from typing import List, Optional

from code_assistant.embedding.models.models import EmbeddingModel
from code_assistant.storage.code_store import (
    CodebaseSnapshot,
    CodeUnit,
    CodeEmbedding,
)


@dataclass
class SearchResult:
    """
    A container for search results that includes both the code unit metadata
    and its similarity score. This provides a clean, type-safe way to return
    search results with their relevance scores.
    """

    code_unit: CodeUnit
    similarity_score: float


class EmbeddingSimilaritySearch:
    def __init__(self, codebase: CodebaseSnapshot, embedding_model: EmbeddingModel):
        """
        Initialize the similarity search engine with code units and their embeddings.
        Each code unit should be a dictionary containing an 'embedding' key with
        the vector representation of that code unit.

        Args:
            codebase: CodebaseSnapshot object containing code units and their embeddings.
            embedding_model: EmbeddingModel object used to generate the embeddings.
        """

        self.codebase = codebase
        self.model = embedding_model

        # Validate code units and embeddings
        self.valid_units = self._validate_code_units()

        # Create and normalize the embedding matrix for efficient computation
        self.embedding_matrix = self._create_embedding_matrix()

        # Store the dimensionality of the embeddings for validation
        self.embedding_dim = (
            self.embedding_matrix.shape[1] if len(self.embedding_matrix) > 0 else 0
        )

    def _validate_code_units(self, enforce_valid: bool = True) -> List[CodeUnit]:
        """
        Validate that all code units have valid embeddings.

        Args:
            enforce_valid: Whether to raise an error for invalid embeddings

        Returns:
            List of code units with valid embeddings

        Raises:
            ValueError: If no valid code units are provided or if embeddings are invalid
        """

        valid_units = []
        for unit in self.codebase.iter_flat():
            if unit.embeddings is not None and self.model.model_name in unit.embeddings:
                valid_units.append(unit)
            elif enforce_valid:
                raise ValueError(
                    f"Code unit {unit.id} does not have a valid embedding for the model {self.model.model_name}"
                )

        if not valid_units:
            raise ValueError("No valid embeddings found in code units")

        # Verify all embeddings have the same dimensionality
        first_dim = len(valid_units[0].embeddings[self.model.model_name])
        if not all(
            len(unit.embeddings[self.model.model_name]) == first_dim
            for unit in valid_units
        ):
            raise ValueError("All embeddings must have the same dimensionality")

        return valid_units

    def _create_embedding_matrix(self) -> np.ndarray:
        """
        Create a normalized matrix of all embeddings for efficient similarity computation.

        Returns:
            2D numpy array where each row is a normalized embedding vector
        """
        # Convert list of embeddings to a 2D numpy array
        embeddings = []
        self.unit_ids = []

        for unit in self.valid_units:
            embeddings.append(unit.embeddings[self.model.model_name].vector)
            self.unit_ids.append(unit.id)

        embedding_matrix = np.array(embeddings)

        # Normalize each embedding to unit length for cosine similarity
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        normalized_matrix = np.divide(embedding_matrix, norms, where=norms != 0)

        return normalized_matrix

    def find_similar(
        self,
        query_embedding: CodeEmbedding,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Find the most similar code units to a query vector using cosine similarity.

        Args:
            query_embedding: The query embedding to compare against
            top_k: Maximum number of results to return
            threshold: Optional similarity threshold (0 to 1) to filter results

        Returns:
            List of SearchResult objects containing matched code units and scores
        """
        if query_embedding.model_name != self.model.model_name:
            raise ValueError(
                f"Query vector was generated with a different model than the code units. "
                f"Query model: {query_embedding.model_name}, Code unit model: {self.model.model_name}"
            )

        if query_embedding.vector.shape != (self.embedding_dim,):
            raise ValueError(
                f"Query vector dimension {query_embedding.vector.shape} does not match "
                f"embedding dimension {self.embedding_dim}"
            )

        # Normalize query vector
        query_norm = np.linalg.norm(query_embedding.vector)
        if query_norm == 0:
            raise ValueError("Query vector cannot be zero")
        normalized_query = query_embedding.vector / query_norm

        # Compute cosine similarities with all embeddings
        similarities = np.dot(self.embedding_matrix, normalized_query)

        # Apply threshold if specified
        if threshold is not None:
            mask = similarities >= threshold
            if not np.any(mask):
                return []
            similarities = similarities[mask]
            valid_indices = np.where(mask)[0]
        else:
            valid_indices = np.arange(len(similarities))

        # Get top-k indices
        top_k = min(top_k, len(valid_indices))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Create search results using unit IDs to fetch code units
        results = []
        for idx in top_indices:
            unit_id = self.unit_ids[valid_indices[idx]]
            code_unit = self.codebase.get_unit_by_id(unit_id)
            results.append(SearchResult(code_unit, float(similarities[idx])))

        return results

    def search_by_type(
        self,
        query_vector: np.ndarray,
        unit_type: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Find similar code units of a specific type (e.g., 'method' or 'class').

        Args:
            query_vector: The query embedding to compare against
            unit_type: Type of code unit to search for (e.g., 'method' or 'class')
            top_k: Maximum number of results to return
            threshold: Optional similarity threshold (0 to 1) to filter results

        Returns:
            List of SearchResult objects containing matched code units and scores
        """
        # Create a mask for the specified type
        type_mask = np.array(self.codebase.get_units_by_type(unit_type), dtype=bool)

        if not np.any(type_mask):
            return []

        # Filter embedding matrix by type
        filtered_matrix = self.embedding_matrix[type_mask]
        filtered_unit_ids = np.array(self.unit_ids)[type_mask]

        # Use the filtered matrix for similarity search
        query_vector = np.array(query_vector)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query vector cannot be zero")
        normalized_query = query_vector / query_norm

        # Compute similarities
        similarities = np.dot(filtered_matrix, normalized_query)

        # Apply threshold if specified
        if threshold is not None:
            mask = similarities >= threshold
            if not np.any(mask):
                return []
            similarities = similarities[mask]
            filtered_unit_ids = filtered_unit_ids[mask]

        # Get top-k results
        top_k = min(top_k, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [
            SearchResult(
                self.codebase.get_unit_by_id(filtered_unit_ids[idx]),
                float(similarities[idx]),
            )
            for idx in top_indices
        ]
