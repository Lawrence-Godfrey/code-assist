from pathlib import Path
from typing import Optional

from code_assistant.embedding.models.models import EmbeddingModelFactory
from code_assistant.logging.logger import LoggingConfig, get_logger
from code_assistant.rag.rag_engine import RAGEngine
from code_assistant.storage.code_store import CodebaseSnapshot

logger = get_logger(__name__)


class RagCommands:
    """Commands for using the RAG engine to answer code-related questions."""

    def prompt(
        self,
        query: str,
        codebase_path: str,
        embedding_model: str = "jinaai/jina-embeddings-v3",
        prompt_model: str = "gpt-4",
        top_k: int = 5,
        threshold: Optional[float] = None,
        logging_enabled: bool = False,
    ) -> None:
        """
        Process a query through the RAG pipeline to generate a contextually-aware response.

        Args:
            query: The question or request about the codebase
            codebase_path: Path to the JSON file containing embedded code units
            embedding_model: Name of the model to use for embeddings
            prompt_model: Name of the LLM to use for response generation
            top_k: Maximum number of similar code units to retrieve
            threshold: Minimum similarity score (0-1) for retrieved code
            logging_enabled: Whether to log detailed pipeline execution info
        """
        LoggingConfig.enabled = logging_enabled

        # Load codebase
        logger.info(f"Loading codebase from {codebase_path}")
        codebase = CodebaseSnapshot.from_json(Path(codebase_path))

        # Initialize embedding model
        embedding_model = EmbeddingModelFactory.create(embedding_model)

        # Initialize RAG engine
        engine = RAGEngine(
            codebase=codebase,
            embedding_model=embedding_model,
            prompt_model=prompt_model,
            top_k=top_k,
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
