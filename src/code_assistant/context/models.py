import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict


class ContextType(str, Enum):
    """Types of contexts supported by the system."""

    CODE = "code"
    DOCUMENT = "document"


class ContextSource(str, Enum):
    """Source types for different contexts."""

    GITHUB = "github"
    NOTION = "notion"
    CONFLUENCE = "confluence"
    PDF = "pdf"


@dataclass
class ContextMetadata:
    """
    Metadata for a single context source.

    This class represents the metadata associated with a specific data source,
    such as a GitHub repository or Notion database.
    """

    type: ContextType
    source: ContextSource
    title: str
    description: str
    source_url: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source.value,
            "title": self.title,
            "description": self.description,
            "source_url": self.source_url,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "additional_metadata": self.additional_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextMetadata":
        """Create metadata instance from dictionary."""
        return cls(
            id=data["id"],
            type=ContextType(data["type"]),
            source=ContextSource(data["source"]),
            title=data["title"],
            description=data["description"],
            source_url=data["source_url"],
            created_at=data["created_at"],
            last_updated=data["last_updated"],
            additional_metadata=data.get("additional_metadata", {}),
        )
