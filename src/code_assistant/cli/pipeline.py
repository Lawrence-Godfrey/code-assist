"""Command line interface for executing an agent task pipeline"""

from code_assistant.logging.logger import LoggingConfig, get_logger
from code_assistant.pipeline.pipeline import Pipeline

logger = get_logger(__name__)


class PipelineCommands:
    """Commands for executing the task pipeline."""

    def start(
        self,
        prompt: str,
        prompt_model: str = "gpt-4",
        logging_enabled: bool = True,
    ) -> None:
        """
        Start the pipeline execution with the given prompt.

        Args:
            prompt: The task prompt to process
            prompt_model: The model to use for LLM prompts
            logging_enabled: Whether to enable detailed logging
        """
        LoggingConfig.enabled = logging_enabled

        logger.info(f"Starting pipeline execution for prompt:\n{prompt}")

        pipeline = Pipeline(prompt_model=prompt_model)
        pipeline.execute(prompt)

        logger.info("Pipeline execution completed successfully")
