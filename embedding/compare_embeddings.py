import json
import os
from dataclasses import dataclass

import fire
import numpy as np
from typing import List, Dict, Optional

from embedding.generate_embeddings import CodeEmbedder


@dataclass
class SearchResult:
    """
    A container for search results that includes both the code unit metadata
    and its similarity score. This provides a clean, type-safe way to return
    search results with their relevance scores.
    """

    code_unit: Dict
    similarity_score: float


class EmbeddingSimilaritySearch:
    def __init__(self, code_units: List[Dict]):
        """
        Initialize the similarity search engine with code units and their embeddings.
        Each code unit should be a dictionary containing an 'embedding' key with
        the vector representation of that code unit.

        Args:
            code_units: List of dictionaries, where each dictionary represents a code unit
                       and must contain an 'embedding' key with its vector representation
        """
        # Store the original code units for reference
        code_units = self._normalise_code_units(code_units)
        self.code_units = self._validate_code_units(code_units)

        # Create and normalize the embedding matrix for efficient computation
        self.embedding_matrix = self._create_embedding_matrix()

        # Store the dimensionality of the embeddings for validation
        self.embedding_dim = (
            self.embedding_matrix.shape[1] if len(self.embedding_matrix) > 0 else 0
        )

    def _normalise_code_units(self, code_units: List[Dict]) -> List[Dict]:

        normalised_units = []

        for unit in code_units:
            normalised_units.append(unit)

            if unit.get("methods"):
                for method in unit["methods"]:
                    normalised_units.append(method)

        return normalised_units

    def _validate_code_units(
        self, code_units: List[Dict], enforce_valid: bool = True
    ) -> List[Dict]:
        """
        Validate that all code units have valid embeddings.

        Args:
            code_units: List of code unit dictionaries to validate
            enforce_valid: Whether to raise an error for invalid embeddings

        Returns:
            List of valid code unit dictionaries

        Raises:
            ValueError: If no valid code units are provided or if embeddings are invalid
        """
        if not code_units:
            raise ValueError("No code units provided")

        valid_units = []
        for unit in code_units:
            if (
                "embedding" in unit
                and isinstance(unit["embedding"], (list, np.ndarray))
                and len(unit["embedding"]) > 0
            ):
                valid_units.append(unit)
            elif enforce_valid:
                raise ValueError(
                    f"Invalid embedding found in code unit: {unit.get('name', 'No name found')}"
                )

        if not valid_units:
            raise ValueError("No valid embeddings found in code units")

        # Verify all embeddings have the same dimensionality
        first_dim = len(valid_units[0]["embedding"])
        if not all(len(unit["embedding"]) == first_dim for unit in valid_units):
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
        for unit in self.code_units:
            embeddings.append(unit["embedding"])

        embedding_matrix = np.array(embeddings)

        # Normalize each embedding to unit length for cosine similarity
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        normalized_matrix = np.divide(embedding_matrix, norms, where=norms != 0)

        return normalized_matrix

    def find_similar(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Find the most similar code units to a query vector using cosine similarity.

        Args:
            query_vector: The query embedding to compare against
            top_k: Maximum number of results to return
            threshold: Optional similarity threshold (0 to 1) to filter results

        Returns:
            List of SearchResult objects containing matched code units and scores
        """
        # Validate query vector
        if not isinstance(query_vector, (np.ndarray, list)):
            raise ValueError("Query vector must be a numpy array or list")
        query_vector = np.array(query_vector)

        if query_vector.shape != (self.embedding_dim,):
            raise ValueError(
                f"Query vector dimension {query_vector.shape} does not match "
                f"embedding dimension {self.embedding_dim}"
            )

        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query vector cannot be zero")
        normalized_query = query_vector / query_norm

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

        # Create search results
        return [
            SearchResult(self.code_units[valid_indices[idx]], float(similarities[idx]))
            for idx in top_indices
        ]

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
        type_mask = np.array(
            [
                unit.get("type", "").lower() == unit_type.lower()
                for unit in self.code_units
            ]
        )

        if not np.any(type_mask):
            return []

        # Filter embedding matrix by type
        filtered_matrix = self.embedding_matrix[type_mask]
        filtered_units = [u for u, m in zip(self.code_units, type_mask) if m]

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
            filtered_units = [filtered_units[i] for i in np.where(mask)[0]]

        # Get top-k results
        top_k = min(top_k, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [
            SearchResult(filtered_units[idx], float(similarities[idx]))
            for idx in top_indices
        ]


def compare(
    input_path: str = "code_units.json",
    output_path: Optional[str] = None,
    query: str = None,
) -> None:
    """
    Generate similarity scores for code units from a JSON file.
    """
    # Convert input path to absolute path if needed
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Please provide the correct path to your code units JSON file."
        )

    # Generate default output path if none provided
    if output_path is None:
        input_dir = os.path.dirname(input_path)
        input_filename = os.path.basename(input_path)
        output_path = os.path.join(input_dir, f"similarities_{input_filename}")

    # Load code units
    print(f"Loading code units from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        code_units = json.load(f)

    # Initialize comparer
    searcher = EmbeddingSimilaritySearch(code_units)

    # Generate query embedding
    embedder = CodeEmbedder()
    query_vector = embedder.generate_embedding(query)

    # Compare all code units
    results = searcher.find_similar(query_vector)

    for result in results:
        print(f"{result.code_unit['filepath']} - Similarity: {result.similarity_score}")

    # Save results
    print("Generating similarity scores...")
    with open(output_path, "w", encoding="utf-8") as f:
        json_results = []
        for result in results:
            json_results.append(
                {
                    "unit": result.code_unit,
                    "similarity": result.similarity_score,
                }
            )
        json.dump(json_results, f, indent=2)


if __name__ == "__main__":
    fire.Fire(compare)
