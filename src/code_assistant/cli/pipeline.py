"""Command line interface for executing the pipeline"""

from typing import Optional

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

        logger.info("Starting pipeline execution")
        logger.info(f"Prompt: {prompt}")

        pipeline = Pipeline()

        try:
            result = pipeline.execute(prompt)
            logger.info("Pipeline execution completed successfully")
            logger.info("Result:")
            logger.info(result)
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
