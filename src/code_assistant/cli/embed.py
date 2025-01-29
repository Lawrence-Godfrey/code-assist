import os
from pathlib import Path
from typing import Optional

from code_assistant.embedding.code_embedder import CodeEmbedder
from code_assistant.embedding.models.models import EmbeddingModelFactory
from code_assistant.logging.logger import get_logger
from code_assistant.storage.stores import CodeStore, JSONCodeStore, MongoDBCodeStore

logger = get_logger(__name__)


class EmbedCommands:
    """Commands for generating and comparing embeddings."""

    def _process_embeddings(
        self,
        codebase: str,
        input_path: str,
        database_url: str,
        model_name: str,
        openai_api_key: Optional[str] = None,
    ) -> None:
        """
        Generate embeddings for code units from a JSON file.

        Args:
            codebase: Name of the codebase to embed.
            input_path: Path to the JSON file containing code units
                             (defaults to 'code_units.json' in current directory)
            database_url: URL for MongoDB database to store code units
            model_name: Name of the Hugging Face model to use for embeddings
            openai_api_key: OpenAI API key for OpenAI models
        """

        code_store = self._setup_code_store(codebase, input_path, database_url)

        model = EmbeddingModelFactory.create(model_name, openai_api_key=openai_api_key)

        # Generate embeddings
        logger.info("Generating embeddings...")
        code_embedder = CodeEmbedder(embedding_model=model)
        units_processed = code_embedder.embed_code_units(code_store)

        code_store.refresh_vector_indexes()

        # Print statistics
        logger.info(f"Total code units processed: {units_processed}")
        logger.info(f"Embedding dimension: {model.embedding_dimension}")

    def generate(
        self,
        codebase: str,
        input_path: str = "code_units.json",
        database_url: str = "mongodb://localhost:27017/",
        model_name: str = EmbeddingModelFactory.get_default_model(),
        openai_api_key: str = os.getenv("OPENAI_API_KEY"),
    ) -> None:
        """Generate embeddings for code units."""

        input_path = os.getenv("CODE_UNITS_PATH") or input_path
        database_url = os.getenv("MONGODB_URL") or database_url

        self._process_embeddings(
            codebase=codebase,
            input_path=input_path,
            database_url=database_url,
            model_name=model_name,
            openai_api_key=openai_api_key,
        )

    def compare(
        self,
        codebase: str,
        query: str,
        input_path: str = "code_units.json",
        database_url: str = "mongodb://localhost:27017/",
        model_name: str = EmbeddingModelFactory.get_default_model(),
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> None:
        """Compare a query against embedded code units."""

        input_path = os.getenv("CODE_UNITS_PATH") or input_path
        database_url = os.getenv("MONGODB_URL") or database_url

        code_store = self._setup_code_store(codebase, input_path, database_url)

        embedding_model = EmbeddingModelFactory.create(model_name)

        query_embedding = embedding_model.generate_embedding(query)

        results = code_store.vector_search(
            query_embedding,
            embedding_model=embedding_model,
            top_k=top_k,
            threshold=threshold,
        )

        for result in results:
            logger.info(
                f"{result.code_unit.fully_qualified_name()} - "
                f"Similarity: {result.similarity_score}"
            )

    def _setup_code_store(
        self,
        codebase: str,
        input_path: str,
        database_url: str,
    ) -> CodeStore:
        if database_url:
            logger.info("Loading code units from MongoDB database")
            code_store = MongoDBCodeStore(
                codebase=codebase, connection_string=database_url
            )
        else:
            input_path = os.path.abspath(input_path)
            if not os.path.exists(input_path):
                raise FileNotFoundError(
                    f"Input file not found: {input_path}\n"
                    "Please provide the correct path to your code units JSON file."
                )

            logger.info(f"Loading code units from {input_path}")
            code_store = JSONCodeStore(codebase=codebase, filepath=Path(input_path))

        if not code_store.codebase_exists():
            raise ValueError(f"Codebase {codebase} does not exist.")

        return code_store
