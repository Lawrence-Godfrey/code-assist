import os
from typing import Optional

from code_assistant.embedding.code_embedder import CodeEmbedder
from code_assistant.logging.logger import get_logger
from code_assistant.models.factory import ModelFactory
from code_assistant.storage.stores.code import MongoDBCodeStore

logger = get_logger(__name__)


class EmbedCommands:
    """Commands for generating and comparing embeddings."""

    def _process_embeddings(
        self,
        codebase: str,
        database_url: str,
        model_name: str,
    ) -> None:
        """
        Generate embeddings for code units from a codebase.

        Args:
            codebase: Name of the codebase to embed.
            database_url: URL for MongoDB database to store code units
            model_name: Name of the Hugging Face model to use for embeddings
        """

        code_store = self._setup_code_store(codebase, database_url)

        embedding_model = ModelFactory.create(model_name)

        # Generate embeddings
        logger.info("Generating embeddings...")
        code_embedder = CodeEmbedder(embedding_model=embedding_model)
        units_processed = code_embedder.embed_code_units(code_store)

        code_store.refresh_vector_indexes()

        # Print statistics
        logger.info(f"Total code units processed: {units_processed}")
        logger.info(f"Embedding dimension: {embedding_model.embedding_dimension}")

    def generate(
        self,
        codebase: str,
        database_url: str = "mongodb://localhost:27017/",
        model_name: str = ModelFactory.get_default_embedding_model(),
    ) -> None:
        """Generate embeddings for code units."""

        database_url = os.getenv("MONGODB_URL") or database_url

        self._process_embeddings(
            codebase=codebase,
            database_url=database_url,
            model_name=model_name,
        )

    def compare(
        self,
        codebase: str,
        query: str,
        database_url: str = "mongodb://localhost:27017/",
        model_name: str = ModelFactory.get_default_embedding_model(),
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> None:
        """Compare a query against embedded code units."""

        database_url = os.getenv("MONGODB_URL") or database_url

        code_store = self._setup_code_store(codebase, database_url)

        embedding_model = ModelFactory.create(model_name)

        query_embedding = embedding_model.generate_embedding(query)

        results = code_store.vector_search(
            query_embedding,
            embedding_model=embedding_model,
            top_k=top_k,
            threshold=threshold,
        )

        for result in results:
            logger.info(
                f"{result.item.fully_qualified_name()} - "
                f"Similarity: {result.similarity_score}"
            )

    def _setup_code_store(
        self,
        codebase: str,
        database_url: str,
    ) -> MongoDBCodeStore:
        logger.info(f"Loading code units from {database_url}...")
        code_store = MongoDBCodeStore(codebase=codebase, connection_string=database_url)

        if not code_store.namespace_exists():
            raise ValueError(f"Codebase {codebase} does not exist.")

        return code_store
