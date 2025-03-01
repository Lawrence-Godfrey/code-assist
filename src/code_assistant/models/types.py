from dataclasses import dataclass
from enum import Enum


class StageStatus(str, Enum):
    """Enum for Stage status values with display names."""
    NOT_STARTED = "not_started" 
    IN_PROGRESS = "in_progress"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def get_display(cls, value):
        """Get the display name for a given status value."""
        display_map = {
            cls.NOT_STARTED: "Not Started",
            cls.IN_PROGRESS: "In Progress",
            cls.WAITING_FOR_APPROVAL: "Waiting for Approval",
            cls.COMPLETED: "Completed",
            cls.FAILED: "Failed"
        }
        return display_map.get(value, value)


class MessageRole(str, Enum):
    """Enum for Message role values."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    @classmethod
    def get_display(cls, value):
        """Get the display name for a given role value."""
        display_map = {
            cls.SYSTEM: "System",
            cls.USER: "User",
            cls.ASSISTANT: "Assistant"
        }
        return display_map.get(value, value)


@dataclass
class Message:
    role: MessageRole
    content: str
