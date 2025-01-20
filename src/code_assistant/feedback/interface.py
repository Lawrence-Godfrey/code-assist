from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol
from .models import FeedbackRequest, FeedbackResponse


class FeedbackObserver(Protocol):
    """Interface for components that need to react to feedback events"""

    def on_feedback_requested(self, request: FeedbackRequest) -> None:
        """Called when feedback is requested"""
        ...

    def on_feedback_received(self, response: FeedbackResponse) -> None:
        """Called when feedback is received."""
        ...

class FeedbackInterface(ABC):
    """Abstract base class for different feedback interfaces."""

    @abstractmethod
    def request_feedback(self, request: FeedbackRequest) -> str:
        """
        Request and collect feedback from the user.

        Args:
            request: The feedback request detailing what feedback is needed

        Returns:
            str: The user's response

        Raises:
            FeedbackCancelled: If the user cancels the feedback request
        """
        pass
