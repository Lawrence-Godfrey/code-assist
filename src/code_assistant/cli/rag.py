import os
from typing import Optional

from code_assistant.logging.logger import LoggingConfig, get_logger
from code_assistant.models.factory import ModelFactory
from code_assistant.rag.rag_engine import RAGEngine
from code_assistant.storage.stores.code import MongoDBCodeStore
from code_assistant.storage.stores.document import MongoDBDocumentStore

logger = get_logger(__name__)


class RagCommands:
    """Commands for using the RAG engine to answer code-related questions."""

    def prompt(
        self,
        query: str,
        codebase: str,
        space_key: str,
        database_url: str = "mongodb://localhost:27017/",
        embedding_model_name: str = ModelFactory.get_default_embedding_model(),
        prompt_model_name: str = ModelFactory.get_default_prompt_model(),
        top_k_code: int = 5,
        top_k_docs: int = 5,
        threshold: Optional[float] = None,
        logging_enabled: bool = False,
    ) -> None:
        """
        Process a query through the RAG pipeline to generate a contextually-aware response.

        Args:
            query: The question or request about the codebase.
            codebase: Name of the codebase containing embedded code units
            space_key: Optional space key for documentation context
            database_url: URL of the database containing embedded code units and docs
            embedding_model_name: Name of the model to use for embeddings
            prompt_model_name: Name of the model to use for response generation
            top_k_code: Maximum number of similar code units to retrieve
            top_k_docs: Maximum number of similar documents to retrieve
            threshold: Minimum similarity score (0-1) for retrieved items
            logging_enabled: Whether to log detailed pipeline execution info
        """
        LoggingConfig.enabled = logging_enabled

        database_url = os.getenv("MONGODB_URL") or database_url

        # Load codebase
        logger.info(f"Loading codebase from database: {database_url}")
        code_store = MongoDBCodeStore(codebase, database_url)

        logger.info(f"Loading documentation from space: {space_key}")
        doc_store = MongoDBDocumentStore(space_key, database_url)

        # Initialize embedding and prompt models
        embedding_model = ModelFactory.create(embedding_model_name)
        prompt_model = ModelFactory.create(prompt_model_name)

        # Initialize RAG engine
        engine = RAGEngine(
            code_store=code_store,
            doc_store=doc_store,
            embedding_model=embedding_model,
            prompt_model=prompt_model,
            top_k_code=top_k_code,
            top_k_docs=top_k_docs,
            threshold=threshold,
        )

        # Process query
        logger.info("\nProcessing query through RAG pipeline...")
        response = engine.process(query)

        # log response
        logger.info("\nResponse:")
        logger.info("=" * 80)
        logger.info(response)
        logger.info("=" * 80)
