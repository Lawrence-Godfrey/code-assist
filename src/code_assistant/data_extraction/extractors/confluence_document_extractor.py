"""
Extract documents from Confluence spaces.

This module provides functionality to extract pages and attachments from Confluence
spaces, process them into Document objects, and store them in MongoDB. I
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from atlassian import Confluence

from code_assistant.logging.logger import get_logger
from code_assistant.storage.document import (
    ConfluencePage
)
from code_assistant.storage.stores.document import MongoDBDocumentStore

logger = get_logger(__name__)


@dataclass
class Space:
    """Represents a Confluence space with its metadata."""

    key: str
    name: str
    description: Optional[str] = None
    homepage_id: Optional[str] = None

    @classmethod
    def from_response(cls, response: dict) -> "Space":
        """Create a Space instance from Confluence API response."""
        return cls(
            key=response["key"],
            name=response["name"],
            description=response.get("description", {}).get("plain", {}).get(
                "value"),
            homepage_id=response.get("homepage", {}).get("id")
        )


class SpaceExistsError(Exception):
    """Raised when a space already exists in the storage."""
    pass


class ConfluenceDocumentExtractor:
    """Extract documents from a Confluence space."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        api_token = os.getenv("CONFLUENCE_API_TOKEN")

        if not api_token:
            raise ValueError(
                "CONFLUENCE_API_TOKEN environment variable not set")

        self.client = Confluence(
            url=self.base_url,
            token=api_token
        )

    async def get_space(self, space_key: str) -> Space:
        """
        Get information about a Confluence space.

        Args:
            space_key: Key of the space to retrieve

        Returns:
            Space object with metadata

        Raises:
            ValueError: If space doesn't exist or can't be accessed
        """
        try:
            response = self.client.get_space(space_key,
                                             expand="description.plain,homepage")
            return Space.from_response(response)
        except Exception as e:
            logger.error(f"Failed to get space {space_key}: {str(e)}")
            raise ValueError(f"Could not access space {space_key}")

    async def _process_content(self, content_id: str, space_key: str) -> \
    Optional[ConfluencePage]:
        """Process a single content item into a ConfluencePage."""
        try:
            # Get full content with body and version info
            content = self.client.get_page_by_id(
                content_id,
                expand="body.storage,version,ancestors"
            )

            # Create page object
            page = ConfluencePage(
                title=content["title"],
                content=content["body"]["storage"]["value"],
                space_key=space_key,
                last_modified=datetime.fromisoformat(
                    content["version"]["when"].replace("Z", "+00:00")
                ),
                version=content["version"]["number"],
                url=f"{self.base_url}/wiki{content['_links']['webui']}",
                parent_id=content["ancestors"][-1]["id"] if content.get(
                    "ancestors") else None
            )

            return page

        except Exception as e:
            logger.error(f"Error processing content {content_id}: {str(e)}")
            return None

    async def extract_documents(
            self,
            space_key: str,
            doc_store: MongoDBDocumentStore,
            overwrite: bool = False,
            limit: Optional[int] = None
    ) -> None:
        """
        Extract all documents from a Confluence space.

        Args:
            space_key: Key of the space to extract from
            doc_store: Document store to save extracted documents
            overwrite: Whether to overwrite existing space content
            limit: Optional limit on number of pages to process

        Raises:
            SpaceExistsError: If space exists and overwrite is False
            ValueError: If space doesn't exist or can't be accessed
        """
        # Verify space exists and get metadata
        space = await self.get_space(space_key)

        # Check if space already exists in storage
        if doc_store.namespace_exists() and not overwrite:
            raise SpaceExistsError(
                f"Space {space_key} already exists. Use overwrite=True to replace."
            )

        if overwrite:
            logger.info(f"Overwriting space {space_key}")
            doc_store.delete_namespace()

        # Process pages
        start = 0
        pages_processed = 0
        while True:
            # Get batch of content
            content_batch = self.client.get_all_pages_from_space(
                space_key,
                start=start,
                limit=25,  # Process in smaller batches
                expand="body.storage,version,ancestors"
            )

            if not content_batch:
                break

            # Process each content item
            for content in content_batch:
                if limit and pages_processed >= limit:
                    break

                if page := await self._process_content(content["id"],
                                                       space_key):
                    doc_store.save_item(page)
                    pages_processed += 1

            if limit and pages_processed >= limit:
                break

            start += len(content_batch)

        logger.info(
            f"Processed {pages_processed} pages from space {space_key} "
            f"({space.name})"
        )
