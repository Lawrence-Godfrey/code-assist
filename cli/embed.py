from pathlib import Path
from typing import Optional, List

from embedding.generate_embeddings import process_embeddings
from embedding.compare_embeddings import EmbeddingSimilaritySearch, CodeEmbedder
from embedding.models.models import EmbeddingModelFactory
from storage.code_store import CodebaseSnapshot


class EmbedCommands:
    """Commands for generating and comparing embeddings."""

    def generate(
        self,
        input_path: str = "code_units.json",
        output_path: Optional[str] = None,
        model_name: str = "jinaai/jina-embeddings-v3",
        openai_api_key: Optional[str] = None,
    ) -> None:
        """Generate embeddings for code units."""
        process_embeddings(
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
