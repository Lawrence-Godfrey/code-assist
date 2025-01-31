"""
Data models for the feedback system.

This module defines the core data structures used throughout the feedback system
to represent feedback requests and responses. It uses dataclasses to provide
immutable, well-structured objects that capture all necessary information about
feedback interactions.

The module provides:
- FeedbackRequest: Represents a request for user feedback with context
- FeedbackResponse: Captures the user's response and links it to the original request

Each model includes automatic timestamp generation and supports optional metadata
for extensibility. The models are designed to be immutable after creation to
ensure data integrity throughout the feedback lifecycle.

Example:
   # Create a feedback request
   request = FeedbackRequest(
       context="requirements_gathering",
       prompt="What is the expected outcome?",
       metadata={"step": "validation"}
   )

   # Create a response to the request
   response = FeedbackResponse(
       request=request,
       response="The system should validate all inputs"
   )

Note:
   Timestamps are automatically generated at instantiation time using
   datetime.now() to provide accurate timing information for feedback
   lifecycle tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


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
