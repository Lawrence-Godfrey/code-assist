import os
from pathlib import Path
from typing import Optional

from code_assistant.embedding.models.models import EmbeddingModelFactory
from code_assistant.logging.logger import LoggingConfig, get_logger
from code_assistant.rag.rag_engine import RAGEngine
from code_assistant.storage.stores import JSONCodeStore, MongoDBCodeStore

logger = get_logger(__name__)


class RagCommands:
    """Commands for using the RAG engine to answer code-related questions."""

    def prompt(
        self,
        query: str,
        codebase: str,
        codebase_path: str = "code_units.json",
        database_url: str = "mongodb://localhost:27017/",
        embedding_model: str = EmbeddingModelFactory.get_default_model(),
        openai_api_key: str = os.getenv("OPENAI_API_KEY"),
        prompt_model: str = "gpt-4",
        top_k: int = 5,
        threshold: Optional[float] = None,
        logging_enabled: bool = False,
    ) -> None:
        """
        Process a query through the RAG pipeline to generate a contextually-aware response.

        Args:
            query: The question or request about the codebase
            codebase: Name of the codebase containing embedded code units
            codebase_path: Path to the JSON file containing embedded code units
            database_url: URL of the database containing embedded code units
            embedding_model: Name of the model to use for embeddings
            openai_api_key: API key for OpenAI models
            prompt_model: Name of the LLM to use for response generation
            top_k: Maximum number of similar code units to retrieve
            threshold: Minimum similarity score (0-1) for retrieved code
            logging_enabled: Whether to log detailed pipeline execution info
        """
        LoggingConfig.enabled = logging_enabled

        codebase_path = os.getenv("CODE_UNITS_PATH") or codebase_path
        database_url = os.getenv("MONGODB_URL") or database_url

        # Load codebase
        if database_url:
            logger.info(f"Loading codebase from database: {database_url}")
            code_store = MongoDBCodeStore(codebase, database_url)
        elif codebase_path:
            logger.info(f"Loading codebase from {codebase_path}")
            code_store = JSONCodeStore(codebase, Path(codebase_path))
        else:
            raise ValueError("Either codebase_path or database_url must be provided.")

        # Initialize embedding model
        embedding_model = EmbeddingModelFactory.create(embedding_model)

        # Initialize RAG engine
        engine = RAGEngine(
            code_store=code_store,
            embedding_model=embedding_model,
            prompt_model=prompt_model,
            top_k=top_k,
            threshold=threshold,
            openai_api_key=openai_api_key,
        )

        # Process query
        logger.info("\nProcessing query through RAG pipeline...")
        response = engine.process(query)

        # log response
        logger.info("\nResponse:")
        logger.info("=" * 80)
        logger.info(response)
        logger.info("=" * 80)
