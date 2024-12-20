import os
from pathlib import Path
from typing import Optional

import fire
from dotenv import load_dotenv

from embedding.models.models import EmbeddingModelFactory
from storage.code_store import CodebaseSnapshot, Class, CodeEmbedding


class CodeEmbedder:

    def __init__(
        self,
        embedding_model: str = "jinaai/jina-embeddings-v3",
        max_length: int = 512,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize code embedder which generates embeddings for code units and queries.

        Args:
            embedding_model (str): Hugging Face model for generating embeddings
            max_length (int): Maximum token length for input sequences (not
                applicable to OpenAIEmbeddingModel)
            openai_api_key (str, optional): OpenAI API key. Required for OpenAI
                models.
        """
        self.model = EmbeddingModelFactory.create(
            embedding_model,
            max_length,
            openai_api_key=openai_api_key,
        )
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


def process_embeddings(
    input_path: str = "code_units.json",
    output_path: Optional[str] = None,
    model_name: str = "jinaai/jina-embeddings-v3",
) -> None:
    """
    Generate embeddings for code units from a JSON file.

    Args:
        input_path (str): Path to the JSON file containing code units
                         (defaults to 'code_units.json' in current directory)
        output_path (str, optional): Path to save the embeddings
                                   (defaults to 'embedded_' + input filename)
        model_name (str): Name of the Hugging Face model to use for embeddings
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

    # Get OpenAI API key if needed
    openai_api_key = None
    if EmbeddingModelFactory.MODEL_REGISTRY.get(model_name) == "OpenAIEmbeddingModel":
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI models"
            )

    # Initialize embedder
    embedder = CodeEmbedder(embedding_model=model_name, openai_api_key=openai_api_key)

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
