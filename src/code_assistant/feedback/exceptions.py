class FeedbackError(Exception):
    """Base class for feedback-related exceptions."""
    pass


class FeedbackCancelled(FeedbackError):
    """Raised when user cancels feedback request."""
    pass
