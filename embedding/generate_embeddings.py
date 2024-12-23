import os
from pathlib import Path
from typing import Optional

import fire
from dotenv import load_dotenv

from embedding.models.models import (
    EmbeddingModelFactory,
    OpenAIEmbeddingModel,
    EmbeddingModel,
)
from storage.code_store import CodebaseSnapshot, Class


class CodeEmbedder:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
    ):
        """
        Initialize code embedder which generates embeddings for code units and queries.

        Args:
            embedding_model (EmbeddingModel): Embedding model to use for generating embeddings
        """
        self.model = embedding_model
        self.embedding_dimension = self.model.embedding_dimension

    def embed_code_units(
        self,
        codebase: CodebaseSnapshot,
    ) -> CodebaseSnapshot:
        """
        Generate embeddings for a list of code units.

        Args:
            codebase (CodebaseSnapshot): Codebase snapshot containing code units

        Returns:
            Updated codebase snapshot with embedded code units.
        """

        for file in codebase:
            for unit in file:
                try:
                    formatted_string = (
                        f"type: {unit.unit_type}, "
                        f"name: {unit.name}, "
                        f"filepath: {unit.file.filepath}, "
                        f"source_code: {unit.source_code}"
                    )
                    unit.embedding = self.model.generate_embedding(formatted_string)

                    if isinstance(unit, Class):
                        for method in unit.methods:
                            formatted_string = (
                                f"type: {method.unit_type}, "
                                f"filepath: {method.class_ref.file.filepath}, "
                                f"class: {method.class_ref.name}, "
                                f"name: {method.name}, "
                                f"source_code: {method.source_code}"
                            )
                            method.embedding = self.model.generate_embedding(
                                formatted_string
                            )

                except Exception as e:
                    print(f"Failed to embed unit {unit.name}: {e}")

        return codebase
