"""
Custom exceptions for the feedback system.

This module defines the exception hierarchy used throughout the feedback system
to handle error cases and user interactions. It provides a base exception class
for all feedback-related errors and specific exceptions for common scenarios
like user cancellation.

"""

class FeedbackError(Exception):
    """Base class for feedback-related exceptions."""
    pass


class FeedbackCancelled(FeedbackError):
    """Raised when user cancels feedback request."""
    pass
