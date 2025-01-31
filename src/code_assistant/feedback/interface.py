"""
Core interfaces for the feedback system.

This module defines the fundamental interfaces that form the backbone of the feedback
system, implementing both the Observer pattern for feedback event handling and the
Strategy pattern for different feedback collection methods.

The module provides:
- A Protocol for observers that want to monitor feedback events
- An abstract base class for implementing different feedback collection interfaces

The observer pattern allows components to monitor and react to feedback events without
tight coupling to the feedback system. The strategy pattern (via FeedbackInterface)
enables different implementations of feedback collection (CLI, web, etc.) while
maintaining a consistent interface.

Example:
    class MyFeedbackMonitor(FeedbackObserver):
        def on_feedback_requested(self, request):
            print(f"Feedback requested: {request.prompt}")

        def on_feedback_received(self, response):
            print(f"Feedback received: {response.response}")

    class WebFeedbackInterface(FeedbackInterface):
        def request_feedback(self, request):
            # Web-based implementation
            ...
"""

from abc import ABC, abstractmethod
from typing import Protocol
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
