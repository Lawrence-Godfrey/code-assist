import os
from pathlib import Path
from typing import Optional

import fire

from embedding.models.models import (
    EmbeddingModelFactory,
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
                    unit.embeddings[self.model.model_name] = self.model.generate_embedding(formatted_string)

                    if isinstance(unit, Class):
                        for method in unit.methods:
                            formatted_string = (
                                f"type: {method.unit_type}, "
                                f"filepath: {method.class_ref.file.filepath}, "
                                f"class: {method.class_ref.name}, "
                                f"name: {method.name}, "
                                f"source_code: {method.source_code}"
                            )
                            method.embeddings[self.model.model_name] = self.model.generate_embedding(
                                formatted_string
                            )

                except Exception as e:
                    print(f"Failed to embed unit {unit.name}: {e}")

        return codebase


def process_embeddings(
    input_path: str = "code_units.json",
    output_path: Optional[str] = None,
    model_name: str = "jinaai/jina-embeddings-v3",
    openai_api_key: Optional[str] = None,
) -> None:
    """
    Generate embeddings for code units from a JSON file.

    Args:
        input_path (str): Path to the JSON file containing code units
                         (defaults to 'code_units.json' in current directory)
        output_path (str, optional): Path to save the embeddings
                                   (defaults to 'embedded_' + input filename)
        model_name (str): Name of the Hugging Face model to use for embeddings
        openai_api_key (str, optional): OpenAI API key for OpenAI models
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
        output_path = os.path.join(input_dir, f"embedded_{input_filename}")

    # Load code units
    print(f"Loading code units from {input_path}")
    codebase = CodebaseSnapshot.from_json(Path(input_path))

    if openai_api_key:
        model = EmbeddingModelFactory.create(model_name, openai_api_key)
    else:
        model = EmbeddingModelFactory.create(model_name)

    # Initialize embedder
    embedder = CodeEmbedder(embedding_model=model)

    # Generate embeddings
    print("Generating embeddings...")
    codebase_with_embeddings = embedder.embed_code_units(codebase)

    # Save results
    print(f"Saving embeddings to {output_path}")
    codebase_with_embeddings.to_json(Path(output_path))

    # Print statistics
    print("\nEmbedding Generation Summary:")
    print(f"Total code units processed: {len(codebase_with_embeddings)}")
    print(f"Embedding dimension: {embedder.embedding_dimension}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(process_embeddings)
