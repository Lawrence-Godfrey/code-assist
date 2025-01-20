from typing import Dict, Any, List, Optional
from .manager import FeedbackManager
from .models import FeedbackRequest


class FeedbackEnabled:
    """Mixin for components that need feedback capabilities."""

    def __init__(self, feedback_manager: FeedbackManager):
        """
        Initialize the feedback capabilities.

        Args:
            feedback_manager: The feedback manager to use
        """
        self._feedback_manager = feedback_manager

    def request_step_feedback(
            self,
            context: str,
            prompt: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Request feedback specific to this pipeline step.

        Args:
            context: The context where feedback is needed
            prompt: The question/prompt for the user
            metadata: Optional additional context-specific data

        Returns:
            The user's response as a string
        """
        request = FeedbackRequest(
            context=context,
            prompt=prompt,
            metadata=metadata
        )

        response = self._feedback_manager.request_feedback(request)
        return response.response
