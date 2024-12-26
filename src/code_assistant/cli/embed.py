import os
from pathlib import Path
from typing import Optional

from code_assistant.embedding.generate_embeddings import CodeEmbedder
from code_assistant.embedding.compare_embeddings import (
    EmbeddingSimilaritySearch,
)
from code_assistant.embedding.models.models import EmbeddingModelFactory
from code_assistant.storage.code_store import CodebaseSnapshot


class EmbedCommands:
    """Commands for generating and comparing embeddings."""

    def _process_embeddings(
        self,
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

    def generate(
        self,
        input_path: str = "code_units.json",
        output_path: Optional[str] = None,
        model_name: str = "jinaai/jina-embeddings-v3",
        openai_api_key: Optional[str] = None,
    ) -> None:
        """Generate embeddings for code units."""
        self._process_embeddings(
            input_path=input_path,
            output_path=output_path,
            model_name=model_name,
            openai_api_key=openai_api_key,
        )

    def compare(
        self,
        query: str,
        input_path: str = "code_units.json",
        model_name: str = "jinaai/jina-embeddings-v3",
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> None:
        """Compare a query against embedded code units."""
        codebase = CodebaseSnapshot.from_json(Path(input_path))
        embedding_model = EmbeddingModelFactory.create(model_name)

        embedder = CodeEmbedder(embedding_model=embedding_model)
        searcher = EmbeddingSimilaritySearch(
            codebase=codebase, embedding_model=embedding_model
        )

        query_embedding = embedder.model.generate_embedding(query)
        results = searcher.find_similar(
            query_embedding=query_embedding, top_k=top_k, threshold=threshold
        )

        for result in results:
            print(
                f"{result.code_unit.fully_qualified_name()} - "
                f"Similarity: {result.similarity_score}"
            )
