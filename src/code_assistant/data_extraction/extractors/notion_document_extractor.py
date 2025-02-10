"""
Extract documents from Notion workspaces.

This module provides functionality to extract pages and content from Notion
workspaces, process them into Document objects, and store them in MongoDB.

For a quick guide on how this works from Notions point of view:
https://developers.notion.com/docs/create-a-notion-integration
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from code_assistant.logging.logger import get_logger
from code_assistant.storage.document import NotionPage
from code_assistant.storage.stores.document import MongoDBDocumentStore

logger = get_logger(__name__)


class SpaceExistsError(Exception):
    """Raised when a space already exists in the storage."""

    pass


class NotionDocumentExtractor:
    """Extract documents from a Notion workspace."""

    def __init__(self):
        """Initialize the Notion document extractor."""
        notion_api_key = os.getenv("NOTION_API_KEY")
        if not notion_api_key:
            raise ValueError(
                "NOTION_API_KEY environment variable not set. "
                "Create an integration at https://www.notion.so/my-integrations"
            )

        from notion_client import Client

        self.client = Client(auth=notion_api_key)

    def validate_database_access(self, database_id: str) -> bool:
        """
        Validate that the database exists and is accessible.

        Args:
            database_id: The 32-character database ID from the Notion URL

        Returns:
            bool: True if database is accessible

        Raises:
            ValueError: If database ID is invalid or database is not accessible
        """
        # Validate ID format
        if not database_id or len(database_id) != 32:
            raise ValueError(
                "Invalid database ID. It should be a 32-character string "
                "found in the database URL: "
                "https://www.notion.so/workspace-name/83c75a51b3aa4d7a867f902df7e3f4e1"
            )

        try:
            # Try to query the database (will fail if no access)
            self.client.databases.retrieve(database_id)
            return True
        except Exception as e:
            raise ValueError(
                f"Cannot access database {database_id}. "
                f"Make sure you've shared it with your integration: {str(e)}"
            )

    async def _process_content(
        self, page_id: str, space_key: str
    ) -> Optional[NotionPage]:
        """Process a single Notion page into a NotionPage."""
        try:
            # Get page content
            page = self.client.pages.retrieve(page_id)

            # Get page content as blocks
            blocks = self.client.blocks.children.list(page_id)
            content = self._blocks_to_markdown(blocks["results"])

            # Create page object
            notion_page = NotionPage(
                title=self._extract_title(page),
                content=content,
                space_key=space_key,
                last_modified=datetime.fromisoformat(
                    page["last_edited_time"].replace("Z", "+00:00")
                ),
                page_id=page_id,
                url=page["url"],
                parent_id=page.get("parent", {}).get("page_id"),
                database_id=page.get("parent", {}).get("database_id"),
            )

            return notion_page

        except Exception as e:
            logger.error(f"Error processing page {page_id}: {str(e)}")
            return None

    def _extract_title(self, page: Dict) -> str:
        """Extract the title from a Notion page."""
        # Get title from properties
        title_prop = page.get("properties", {}).get("title", {})
        if title_prop:
            title_content = title_prop.get("title", [])
            if title_content:
                return title_content[0].get("plain_text", "Untitled")
        return "Untitled"

    def _blocks_to_markdown(self, blocks: List[Dict]) -> str:
        """Convert Notion blocks to markdown format."""
        markdown = []

        for block in blocks:
            block_type = block["type"]
            if block_type == "paragraph":
                text = self._get_rich_text(block["paragraph"]["rich_text"])
                markdown.append(text + "\n\n")
            elif block_type == "heading_1":
                text = self._get_rich_text(block["heading_1"]["rich_text"])
                markdown.append(f"# {text}\n\n")
            elif block_type == "heading_2":
                text = self._get_rich_text(block["heading_2"]["rich_text"])
                markdown.append(f"## {text}\n\n")
            elif block_type == "heading_3":
                text = self._get_rich_text(block["heading_3"]["rich_text"])
                markdown.append(f"### {text}\n\n")
            elif block_type == "bulleted_list_item":
                text = self._get_rich_text(block["bulleted_list_item"]["rich_text"])
                markdown.append(f"* {text}\n")
            elif block_type == "numbered_list_item":
                text = self._get_rich_text(block["numbered_list_item"]["rich_text"])
                markdown.append(f"1. {text}\n")
            elif block_type == "code":
                code = block["code"]["rich_text"][0]["plain_text"]
                language = block["code"]["language"]
                markdown.append(f"```{language}\n{code}\n```\n\n")
            elif block_type == "quote":
                text = self._get_rich_text(block["quote"]["rich_text"])
                markdown.append(f"> {text}\n\n")

        return "".join(markdown)

    def _get_rich_text(self, rich_text: List[Dict]) -> str:
        """Extract plain text from Notion's rich text format."""
        return "".join(text["plain_text"] for text in rich_text)

    async def extract_documents(
        self,
        database_id: str,
        doc_store: MongoDBDocumentStore,
        overwrite: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        """
        Extract all documents from a Notion database.

        Args:
            database_id: ID of the Notion database to extract from
            doc_store: Document store to save extracted documents
            overwrite: Whether to overwrite existing space content
            limit: Optional limit on number of pages to process

        Raises:
            SpaceExistsError: If space exists and overwrite is False
            ValueError: If database doesn't exist or can't be accessed
        """
        # Use database_id as space_key for consistency
        space_key = database_id

        # Check if space already exists in storage
        if doc_store.namespace_exists() and not overwrite:
            raise SpaceExistsError(
                f"Space {space_key} already exists. Use overwrite=True to replace."
            )

        if overwrite:
            logger.info(f"Overwriting space {space_key}")
            doc_store.delete_namespace()

        # Validate database access first
        self.validate_database_access(database_id)

        try:
            # Query database
            pages_processed = 0
            has_more = True
            next_cursor = None

            while has_more:
                # Get batch of pages
                query_response = self.client.databases.query(
                    database_id=database_id,
                    start_cursor=next_cursor,
                    page_size=min(100, limit - pages_processed if limit else 100),
                )

                # Process each page
                for page in query_response["results"]:
                    if limit and pages_processed >= limit:
                        break

                    if page := await self._process_content(page["id"], space_key):
                        doc_store.save_item(page)
                        pages_processed += 1

                if limit and pages_processed >= limit:
                    break

                # Update pagination info
                has_more = query_response["has_more"]
                next_cursor = query_response.get("next_cursor")

            logger.info(
                f"Processed {pages_processed} pages from database {database_id}"
            )

        except Exception as e:
            logger.error(f"Error extracting from Notion: {str(e)}")
            raise ValueError(f"Could not access database {database_id}: {str(e)}")
