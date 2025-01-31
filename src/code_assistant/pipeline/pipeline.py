"""
This module implements a pipeline for executing tasks, as requested by the user,
in a defined set of steps. It uses the Chain of Responsibility pattern to process
through multiple stages.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from code_assistant.feedback.interface import FeedbackInterface
from code_assistant.feedback.interfaces.cli import CLIFeedbackInterface
from code_assistant.feedback.manager import FeedbackManager
from code_assistant.logging.logger import get_logger
from code_assistant.pipeline.requirements_gathering.step import RequirementsGatherer
from code_assistant.pipeline.step import PipelineStep
from code_assistant.prompt.models import PromptModel

logger = get_logger(__name__)


class Pipeline(ABC):
    """
    Base pipeline class that provides the framework for executing steps in sequence.
    Uses the Chain of Responsibility pattern for step execution.
    """

    def __init__(
        self,
        prompt_model: PromptModel,
        feedback_interface: Optional[FeedbackInterface] = None,
    ) -> None:
        """
        Initialize the base pipeline.

        Args:
            prompt_model (PromptModel): Prompt model to use.
            feedback_interface: Optional custom feedback interface. If None,
                defaults to CLIFeedbackInterface
        """
        # Initialize feedback system
        if feedback_interface is None:
            feedback_interface = CLIFeedbackInterface()

        self._prompt_model = prompt_model
        self._feedback_manager = FeedbackManager(interface=feedback_interface)
        self._steps: List[PipelineStep] = []

    @property
    def steps(self) -> List[PipelineStep]:
        """Get the list of pipeline steps."""
        return self._steps

    def _chain_steps(self) -> None:
        """Chain the steps together using the Chain of Responsibility pattern."""
        if not self._steps:
            return

        # Link each step to the next one
        for i in range(len(self._steps) - 1):
            self._steps[i].set_next(self._steps[i + 1])

    @abstractmethod
    def _initialize_steps(self) -> None:
        """Initialize the pipeline steps. Must be implemented by subclasses."""
        pass

    def execute(self, prompt: str) -> Dict:
        """
        Execute the full pipeline.

        Args:
            prompt: The user's task prompt

        Returns:
            Dict containing the pipeline execution results

        Raises:
            ValueError: If pipeline has no steps or if any step fails
        """
        # Initialize steps if not already done
        if not self._steps:
            self._initialize_steps()
            self._chain_steps()

        if not self._steps:
            raise ValueError("Pipeline has no steps configured")

        context = {"prompt": prompt}

        try:
            # Start execution with the first step
            logger.info(f"Starting pipeline execution with {len(self._steps)} steps")
            self._steps[0].execute(context)
            return context
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise


class AgentPipeline(Pipeline):
    """
    Concrete implementation of the pipeline for agent-based task execution.
    Currently, implements requirements gathering with planned expansion for
    additional steps.
    """

    def _initialize_steps(self) -> None:
        """Initialize the agent pipeline steps."""
        # Initialize requirements gathering step
        requirements_gatherer = RequirementsGatherer(
            prompt_model=self._prompt_model, feedback_manager=self._feedback_manager
        )

        # Configure pipeline steps
        self._steps = [
            requirements_gatherer,
            # Additional steps will be added here as they are implemented
        ]
