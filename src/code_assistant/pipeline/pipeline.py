"""
This module implements a pipeline for executing tasks, as requested by the user,
in a defined set of steps. It uses the Chain of Responsibility pattern to process
through multiple stages.
"""
from typing import Dict, Optional

from code_assistant.logging.logger import get_logger
from code_assistant.pipeline.requirements_engineering.step import RequirementsEngineering

logger = get_logger(__name__)


class Pipeline:
    """
    Main pipeline class that orchestrates the execution of all steps.
    """

    def __init__(self) -> None:
        """Initialise the pipeline with the defined steps."""
        self.requirements_step = RequirementsEngineering()

        # Currently we only have one step, but we'll add more and chain them
        # together as we implement them
        self._first_step = self.requirements_step

    def execute(self, prompt: str) -> Dict:
        """
        Execute the full pipeline.

        Args:
            prompt: The user's task prompt

        Returns:
            Dict containing the pipeline execution results.

        Raises:
            ValueError: If any step fails
        """
        context = {"prompt": prompt}

        try:
            logger.info("Beginning pipeline execution")
            self._first_step.execute(context)
            return context
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
