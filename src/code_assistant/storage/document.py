"""
Core document models for storing and managing documentation content.

This module defines the fundamental data structures for representing documents
and their embeddings, following similar patterns to the code storage system
while maintaining simplicity and separation of concerns.
"""

import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from code_assistant.logging.logger import get_logger
from code_assistant.storage.types import EmbeddingUnit

logger = get_logger(__name__)


@dataclass
class Document(ABC):
    """
    Abstract base class for all document types (pages, attachments, etc.).
    Provides common attributes and functionality for all document types.
    """

    title: str
    content: str
    space_key: str
    last_modified: datetime
    doc_type: str = field(init=False)  # Set by subclasses
    embeddings: Dict[str, EmbeddingUnit] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert the document to a dictionary."""
        return {
            "id": self.id,
            "doc_type": self.doc_type,
            "title": self.title,
            "content": self.content,
            "space_key": self.space_key,
            "last_modified": self.last_modified.isoformat(),
            "embeddings": {
                model_name: embedding.to_dict()
                for model_name, embedding in self.embeddings.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Create a Document from a dictionary."""
        doc_type = data.pop("doc_type")
        if doc_type == "page":
            return ConfluencePage.from_dict(data)
        else:
            raise ValueError(f"Invalid document type: {doc_type}")

    def get_full_text(self) -> str:
        """
        Get the full text representation of the document for embedding.
        This combines title and content with appropriate weighting.
        """
        return f"Title: {self.title}\n\nContent: {self.content}"


@dataclass
class ConfluencePage(Document):
    """Represents a Confluence page with its specific metadata."""

    doc_type = "page"
    parent_id: Optional[str] = None
    version: int = 1
    url: str = ""

    def to_dict(self) -> dict:
        """Convert the Confluence page to a dictionary."""
        result = super().to_dict()
        result.update(
            {"parent_id": self.parent_id, "version": self.version, "url": self.url}
        )
        return result

    # TODO: Deal with duplication here
    @classmethod
    def from_dict(cls, data: dict) -> "ConfluencePage":
        """Create a ConfluencePage from a dictionary."""
        # Convert ISO format string back to datetime
        data["last_modified"] = datetime.fromisoformat(data["last_modified"])

        # Extract embeddings data
        embeddings_data = data.pop("embeddings", {})

        # Create the page instance
        page = cls(**data)

        # Restore embeddings
        page.embeddings = {
            model_name: EmbeddingUnit.from_dict(embedding_data)
            for model_name, embedding_data in embeddings_data.items()
        }

        return page
