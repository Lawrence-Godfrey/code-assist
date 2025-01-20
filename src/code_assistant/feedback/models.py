from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass
class FeedbackRequest:
    """
    Represents a request for feedback from any system component.

    Attributes:
        context: The context where feedback is needed (e.g. requirements_gathering)
        prompt: The question / prompt for the user
        metadata: Optional additional context-specific metadata
    """
    context: str
    prompt: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now())


@dataclass
class FeedbackResponse:
    """
    Represents the response to a feedback request.

    Attributes:
        request: The original feedback request
        response: The feedback provided by the user
        timestamp: When the feedback was received
    """
    request: FeedbackRequest
    response: str
    timestamp: datetime = field(default_factory=lambda: datetime.now())
