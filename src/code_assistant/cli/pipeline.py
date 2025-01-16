"""Command line interface for executing an agent task pipeline"""

from code_assistant.logging.logger import LoggingConfig, get_logger
from code_assistant.pipeline.pipeline import Pipeline

logger = get_logger(__name__)


class PipelineCommands:
    """Commands for executing the task pipeline."""

    def start(
        self,
        prompt: str,
        logging_enabled: bool = True,
    ) -> None:
        """
        Start the pipeline execution with the given prompt.

        Args:
            prompt: The task prompt to process
            logging_enabled: Whether to enable detailed logging
        """
        LoggingConfig.enabled = logging_enabled

        logger.info(f"Starting pipeline execution for prompt:\n{prompt}")

        pipeline = Pipeline()

        try:
            result = pipeline.execute(prompt)
            logger.info("Pipeline execution completed successfully")
            logger.info(f"Result:\n{result}")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
