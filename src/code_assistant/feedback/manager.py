"""
Central orchestrator for the feedback system.

This module implements the core feedback management system using the Observer pattern,
coordinating feedback collection and event notifications. The FeedbackManager acts as
both a central coordinator and an observer itself, managing the feedback lifecycle and
notifying other system components of feedback events.

The manager handles:
- Coordinating feedback requests through configurable interfaces
- Managing observer subscriptions for feedback events
- Maintaining the feedback lifecycle (request -> collection -> notification)
- Creating and distributing feedback response objects

The system is designed to be extensible, allowing different feedback interfaces
(CLI, web, etc.) to be plugged in while maintaining a consistent event notification
system for all observers.

Example:
    # Initialize with CLI interface
    manager = FeedbackManager(CLIFeedbackInterface())

    # Add observers
    manager.add_observer(MetricsCollector())
    manager.add_observer(Logger())

    # Request feedback
    response = manager.request_feedback(
        FeedbackRequest(context="user_input", prompt="Your name?")
    )

Note:
    FeedbackManager itself implements FeedbackObserver, allowing it to be part
    of larger feedback chains if needed.
"""

from typing import List
from .interface import FeedbackInterface, FeedbackObserver
from .models import FeedbackRequest, FeedbackResponse

class FeedbackManager:
    """
    Orchestrates feedback collection and processing

    This class manages the feedback lifecycle, including:
    - Collecting feedback through the specified interface
    - Notifying observers of feedback events
    """

    def __init__(self, interface: FeedbackInterface):
        """
        Initialize the feedback manager

        Args:
            interface: The interface to use for collecting feedback
        """
        self._interface = interface
        self._observers: List[FeedbackObserver] = []

    def add_observer(self, observer: FeedbackObserver) -> None:
        """Add an observer to be notified of feedback events."""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: FeedbackObserver) -> None:
        """Remove an observer from the notification list."""
        if observer in self._observers:
            self._observers.remove(observer)

    def request_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """
        Request and process feedback.

        Args:
            request: The feedback request

        Returns:
            The feedback response containing the user's input
        """

        # Notify observers that feedback is being requested
        for observer in self._observers:
            observer.on_feedback_requested(request)

        # Collect feedback through the interface
        response_text = self._interface.request_feedback(request)

        # Create response object
        response = FeedbackResponse(request=request, response=response_text)

        # Notify observers of the received feedback
        for observer in self._observers:
            observer.on_feedback_received(response)

        return response