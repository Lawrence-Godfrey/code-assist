"""
Mixin class for enabling feedback capabilities in components.

This module provides a mixin class that can be used to add feedback functionality
to any component in the system. It simplifies the feedback request process by
providing a clean, high-level interface that hides the complexity of feedback
request creation and management.

The mixin pattern is used here to:
- Add feedback capabilities without inheritance conflicts
- Provide a consistent interface across different components
- Simplify feedback request creation and handling
- Enable code reuse for feedback functionality

Example:
   class MyPipelineStep(PipelineStep, FeedbackEnabled):
       def __init__(self, feedback_manager):
           FeedbackEnabled.__init__(self, feedback_manager)

       def execute(self):
           response = self.request_step_feedback(
               context="validation",
               prompt="Please confirm these changes",
               metadata={"step": "validation"}
           )
           # Process response...

Note:
   Components using this mixin should initialize it properly by calling
   FeedbackEnabled.__init__() with a FeedbackManager instance.
"""

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
