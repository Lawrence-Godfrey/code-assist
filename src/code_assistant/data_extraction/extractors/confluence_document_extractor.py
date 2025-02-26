"""
Extract documents from Confluence spaces.

This module provides functionality to extract pages and attachments from Confluence
spaces, process them into Document objects, and store them in MongoDB. I
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from atlassian import Confluence
from html2markdown import convert

from code_assistant.logging.logger import get_logger
from code_assistant.storage.document import ConfluencePage
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
            description=response.get("description", {}).get("plain", {}).get("value"),
            homepage_id=response.get("homepage", {}).get("id"),
        )


class SpaceExistsError(Exception):
    """Raised when a space already exists in the storage."""

    pass


class ConfluenceDocumentExtractor:
    """Extract documents from a Confluence space."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        confluence_username = os.getenv("CONFLUENCE_USERNAME")
        confluence_api_token = os.getenv("CONFLUENCE_API_TOKEN")

        if not confluence_username:
            raise ValueError("CONFLUENCE_USERNAME environment variable not set")

        if not confluence_api_token:
            raise ValueError("CONFLUENCE_API_TOKEN environment variable not set")

        self.client = Confluence(
            url=self.base_url,
            username=confluence_username,
            password=confluence_api_token,
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
            response = self.client.get_space(
                space_key, expand="description.plain,homepage"
            )
            return Space.from_response(response)
        except Exception as e:
            logger.error(f"Failed to get space {space_key}: {str(e)}")
            raise ValueError(f"Could not access space {space_key}")

    async def _process_content(
        self, content_id: str, space_key: str
    ) -> Optional[ConfluencePage]:
        """Process a single content item into a ConfluencePage."""
        try:
            # Get full content with body and version info
            content = self.client.get_page_by_id(
                content_id, expand="body.storage,version,ancestors"
            )

            # Convert HTML content to markdown
            html_content = content["body"]["storage"]["value"]
            cleaned_content = self._clean_confluence_markup(html_content)
            markdown_content = convert(cleaned_content)

            # Create page object
            page = ConfluencePage(
                title=content["title"],
                content=markdown_content,
                space_key=space_key,
                last_modified=datetime.fromisoformat(
                    content["version"]["when"].replace("Z", "+00:00")
                ),
                version=content["version"]["number"],
                url=f"{self.base_url}/wiki{content['_links']['webui']}",
                parent_id=(
                    content["ancestors"][-1]["id"] if content.get("ancestors") else None
                ),
            )

            return page

        except Exception as e:
            logger.error(f"Error processing content {content_id}: {str(e)}")
            return None

    def _clean_confluence_markup(self, content: str) -> str:
        """
        Clean Confluence-specific markup from content.

        Args:
            content: Raw content from Confluence API

        Returns:
            Cleaned content with macros replaced or removed
        """
        # Dictionary of macro handlers
        macro_handlers = {
            "panel": lambda body, params: f"\n> {body}\n",
            "info": lambda body, params: f"\n**Info:** {body}\n",
            "note": lambda body, params: f"\n**Note:** {body}\n",
            "warning": lambda body, params: f"\n**Warning:** {body}\n",
            "code": lambda body, params: f"\n```{params.get('language', '')}\n{body}\n```\n",
            "noformat": lambda body, params: f"\n```\n{body}\n```\n",
            # Remove interactive/dynamic elements completely
            "livesearch": lambda body, params: "",
            "recently-updated": lambda body, params: "",
        }

        def extract_macro_content(match) -> str:
            """Extract and format content from a Confluence macro."""
            macro_name = re.search(r'ac:name="([^"]+)"', match.group(0))
            if not macro_name:
                return ""

            macro_name = macro_name.group(1)

            # Extract parameters
            params = {}
            param_matches = re.finditer(
                r'ac:parameter ac:name="([^"]+)">([^<]+)', match.group(0)
            )
            for param_match in param_matches:
                params[param_match.group(1)] = param_match.group(2)

            # Extract body content
            body = ""
            body_match = re.search(
                r"<ac:rich-text-body>(.*?)</ac:rich-text-body>",
                match.group(0),
                re.DOTALL,
            )
            if body_match:
                body = body_match.group(1)
                # Clean up any nested HTML
                body = re.sub(r"<[^>]+>", "", body)
                body = body.strip()

            # Handle the macro using appropriate handler
            handler = macro_handlers.get(macro_name, lambda body, params: body)
            return handler(body, params)

        def clean_content(content: str) -> str:
            # Remove Confluence structural elements
            content = re.sub(r"<ri:.*?</ri:.*?>", "", content)

            # Handle structured macros
            content = re.sub(
                r"<ac:structured-macro.*?</ac:structured-macro>",
                extract_macro_content,
                content,
                flags=re.DOTALL,
            )

            # Handle other Confluence-specific elements
            content = re.sub(r"<ac:link>.*?</ac:link>", "", content, flags=re.DOTALL)
            content = re.sub(r"<ac:image>.*?</ac:image>", "", content, flags=re.DOTALL)

            # Clean up any remaining Confluence tags
            content = re.sub(r"<ac:.*?>", "", content)
            content = re.sub(r"</ac:.*?>", "", content)

            return content.strip()

        return clean_content(content)

    async def extract_documents(
        self,
        space_key: str,
        doc_store: MongoDBDocumentStore,
        overwrite: bool = False,
        limit: Optional[int] = None,
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
                expand="body.storage,version,ancestors",
            )

            if not content_batch:
                break

            # Process each content item
            for content in content_batch:
                if limit and pages_processed >= limit:
                    break

                if page := await self._process_content(content["id"], space_key):
                    doc_store.save_item(page)
                    pages_processed += 1

            if limit and pages_processed >= limit:
                break

            start += len(content_batch)

        logger.info(
            f"Processed {pages_processed} pages from space {space_key} "
            f"({space.name})"
        )
