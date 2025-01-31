"""
This module implements a pipeline for executing tasks, as requested by the user,
in a defined set of steps. It uses the Chain of Responsibility pattern to process
through multiple stages.
"""
from typing import Dict, Optional

from code_assistant.feedback.manager import FeedbackManager
from code_assistant.feedback.interface import FeedbackInterface
from code_assistant.feedback.interfaces.cli import CLIFeedbackInterface
from code_assistant.logging.logger import get_logger
from code_assistant.prompt.models import PromptModel
from code_assistant.pipeline.requirements_gathering.step import RequirementsGatherer


logger = get_logger(__name__)


class Pipeline:
    """
    Main pipeline class that orchestrates the execution of all steps.
    Includes feedback system integration for interactive task processing.
    """

    def __init__(
        self,
        prompt_model: PromptModel,
        feedback_interface: Optional[FeedbackInterface] = None,
    ) -> None:
        """
        Initialise the pipeline with the defined steps.

        Args:
            prompt_model: The model to use for LLM prompts
            feedback_interface: Optional custom feedback interface. If None,
                defaults to CLIFeedbackInterface
        """
        # Initialize feedback system
        if feedback_interface is None:
            feedback_interface = CLIFeedbackInterface()

        self._feedback_manager = FeedbackManager(interface=feedback_interface)

        # Initialize pipeline steps
        self.requirements_step = RequirementsGatherer(
            prompt_model=prompt_model,
            feedback_manager=self._feedback_manager
        )

        # Set up the pipeline chain
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
