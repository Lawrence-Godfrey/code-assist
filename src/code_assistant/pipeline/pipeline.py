"""
This module implements a pipeline for executing tasks, as requested by the user,
in a defined set of steps. It uses the Chain of Responsibility pattern to process
through multiple stages.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

from code_assistant.logging.logger import get_logger

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
            self._first_step.execute(context)
            return context
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    def __init__(self) -> None:
        self._next_step: Optional[PipelineStep] = None

    def set_next(self, step: 'PipelineStep') -> 'PipelineStep':
        """Set the next step in the pipeline."""
        self._next_step = step
        return step

    def execute_next(self, context: Dict) -> None:
        """Execute the next step in the pipeline if it exists."""
        if self._next_step:
            return self._next_step.execute(context)
        return None

    @abstractmethod
    def execute(self, context: Dict) -> None:
        """Execute this pipeline step."""
        pass