"""
Core context management system.

This module defines the fundamental data structures and interfaces for the context
management system, which provides a unified way to handle multiple data sources
(code, documents) while maintaining clean separation of concerns.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection

from code_assistant.context.models import ContextMetadata, ContextSource, ContextType
from code_assistant.data_extraction.extractors.confluence_document_extractor import (
    ConfluenceDocumentExtractor,
)
from code_assistant.data_extraction.extractors.github_code_extractor import (
    GitHubCodeExtractor,
)
from code_assistant.data_extraction.extractors.local_pdf_document_extractor import (
    PDFDocumentExtractor,
)
from code_assistant.data_extraction.extractors.notion_document_extractor import (
    NotionDocumentExtractor,
)
from code_assistant.logging.logger import get_logger
from code_assistant.storage.stores.code import MongoDBCodeStore
from code_assistant.storage.stores.document import MongoDBDocumentStore

logger = get_logger(__name__)


class ContextRegistry:
    """
    Manages context metadata and lifecycle.

    This class handles the storage and retrieval of context metadata, providing
    a centralized registry for all data sources in the system.
    """

    def __init__(self, database_url: str):
        """
        Initialize the context registry.

        Args:
            database_url: MongoDB connection URL
        """
        self.client = MongoClient(database_url)
        self.db = self.client.code_assistant
        self._init_collections()

    def _init_collections(self) -> None:
        """Initialize MongoDB collections with proper indexes."""
        self.contexts: Collection = self.db.contexts

        # Ensure indexes
        self.contexts.create_index([("id", ASCENDING)], unique=True)
        self.contexts.create_index([("type", ASCENDING)])
        self.contexts.create_index([("source", ASCENDING)])
        self.contexts.create_index([("created_at", ASCENDING)])

    def add_context(self, metadata: ContextMetadata) -> None:
        """
        Add new context to registry.

        Args:
            metadata: Context metadata to store

        Raises:
            ValueError: If context with same ID already exists
        """
        if self.get_context(metadata.id):
            raise ValueError(f"Context with ID {metadata.id} already exists")

        self.contexts.insert_one(metadata.to_dict())
        logger.info(f"Added new context: {metadata.title} ({metadata.id})")

    def remove_context(self, context_id: str) -> None:
        """
        Remove context from registry.

        Args:
            context_id: ID of context to remove

        Raises:
            ValueError: If context doesn't exist
        """
        if not self.get_context(context_id):
            raise ValueError(f"Context {context_id} does not exist")

        self.contexts.delete_one({"id": context_id})
        logger.info(f"Removed context: {context_id}")

    def get_context(self, context_id: str) -> Optional[ContextMetadata]:
        """
        Get context metadata by ID.

        Args:
            context_id: ID of context to retrieve

        Returns:
            ContextMetadata if found, None otherwise
        """
        if doc := self.contexts.find_one({"id": context_id}):
            return ContextMetadata.from_dict(doc)
        return None

    def list_contexts(self) -> List[ContextMetadata]:
        """Get all registered contexts."""
        return [ContextMetadata.from_dict(doc) for doc in self.contexts.find()]

    def update_context(self, context_id: str, **updates) -> None:
        """
        Update context metadata fields.

        Args:
            context_id: ID of context to update
            **updates: Field updates as keyword arguments

        Raises:
            ValueError: If context doesn't exist
        """
        if not (context := self.get_context(context_id)):
            raise ValueError(f"Context {context_id} does not exist")

        # Update only valid fields
        valid_updates = {}
        for field, value in updates.items():
            if hasattr(context, field):
                valid_updates[field] = value

        if valid_updates:
            valid_updates["last_updated"] = datetime.now()
            self.contexts.update_one({"id": context_id}, {"$set": valid_updates})
            logger.info(f"Updated context {context_id}: {valid_updates}")


class ContextManager:
    """
    Central manager for context operations.

    This class coordinates all context-related operations, including data
    extraction, storage management, and metadata tracking.
    """

    def __init__(self, database_url: str):
        """
        Initialize the context manager.

        Args:
            database_url: MongoDB connection URL
        """
        self.database_url = database_url
        self.registry = ContextRegistry(database_url)

    def _get_extractor(self, source: ContextSource, **kwargs):
        """Get appropriate extractor for source type."""
        extractors = {
            ContextSource.GITHUB: lambda: GitHubCodeExtractor(**kwargs),
            ContextSource.NOTION: lambda: NotionDocumentExtractor(),
            ContextSource.CONFLUENCE: lambda: ConfluenceDocumentExtractor(
                base_url=kwargs.get("base_url", "")
            ),
            ContextSource.PDF: lambda: PDFDocumentExtractor(),
        }

        if source not in extractors:
            raise ValueError(f"Unsupported source type: {source}")

        return extractors[source]()

    def _get_store(self, type: ContextType, namespace: str):
        """Get appropriate store for context type."""
        stores = {
            ContextType.CODE: lambda: MongoDBCodeStore(
                codebase=namespace, connection_string=self.database_url
            ),
            ContextType.DOCUMENT: lambda: MongoDBDocumentStore(
                source_id=namespace, connection_string=self.database_url
            ),
        }

        if type not in stores:
            raise ValueError(f"Unsupported context type: {type}")

        return stores[type]()

    async def add_context(
        self,
        type: ContextType,
        source: ContextSource,
        title: str,
        description: str,
        source_url: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
        **extraction_kwargs,
    ) -> str:
        """
        Add new context from source.

        Args:
            type: Type of context (code/document)
            source: Source type (github/notion/etc)
            title: Human-readable title
            description: Context description
            source_url: URL or path to source
            additional_metadata: Optional additional metadata
            **extraction_kwargs: Additional kwargs for extractor

        Returns:
            ID of created context

        Raises:
            ValueError: If extraction fails
        """
        # Create metadata
        metadata = ContextMetadata(
            type=type,
            source=source,
            title=title,
            description=description,
            source_url=source_url,
            additional_metadata=additional_metadata or {},
        )

        try:
            # Initialize extractor and store
            extractor = self._get_extractor(source, **extraction_kwargs)
            store = self._get_store(type, metadata.id)

            # Extract data
            if source == ContextSource.GITHUB:
                repo = extractor.clone_repository(source_url)
                extractor.process_repository(repo, store)
            elif source == ContextSource.NOTION:
                await extractor.extract_documents(
                    database_id=extraction_kwargs["database_id"], doc_store=store
                )
            elif source == ContextSource.CONFLUENCE:
                await extractor.extract_documents(
                    space_key=extraction_kwargs["space_key"], doc_store=store
                )
            elif source == ContextSource.PDF:
                await extractor.extract_documents(
                    path=source_url, doc_store=store, collection_name=metadata.id
                )

            # Store metadata
            self.registry.add_context(metadata)

            return metadata.id

        except Exception as e:
            logger.error(f"Failed to add context: {str(e)}")
            raise ValueError(f"Context extraction failed: {str(e)}")

    async def refresh_context(
        self, context_id: Optional[str] = None, all: bool = False
    ) -> None:
        """
        Refresh one or all contexts.

        Args:
            context_id: ID of context to refresh (None if all=True)
            all: Whether to refresh all contexts

        Raises:
            ValueError: If context doesn't exist or refresh fails
        """
        contexts = []
        if all:
            contexts = self.registry.list_contexts()
        elif context_id:
            if context := self.registry.get_context(context_id):
                contexts = [context]
            else:
                raise ValueError(f"Context {context_id} does not exist")
        else:
            raise ValueError("Must specify context_id or all=True")

        for context in contexts:
            try:
                # Remove existing data
                store = self._get_store(context.type, context.id)
                store.delete_namespace()

                # Re-extract data
                extraction_kwargs = context.additional_metadata.copy()
                extraction_kwargs.update({"overwrite": True})

                extractor = self._get_extractor(context.source, **extraction_kwargs)

                if context.source == ContextSource.GITHUB:
                    repo = extractor.clone_repository(context.source_url)
                    extractor.process_repository(repo, store)
                elif context.source == ContextSource.NOTION:
                    await extractor.extract_documents(
                        database_id=extraction_kwargs["database_id"], doc_store=store
                    )
                elif context.source == ContextSource.CONFLUENCE:
                    await extractor.extract_documents(
                        space_key=extraction_kwargs["space_key"], doc_store=store
                    )
                elif context.source == ContextSource.PDF:
                    await extractor.extract_documents(
                        path=context.source_url,
                        doc_store=store,
                        collection_name=context.id,
                    )

                # Update metadata
                self.registry.update_context(context.id, last_updated=datetime.now())

                logger.info(f"Refreshed context: {context.title} ({context.id})")

            except Exception as e:
                logger.error(f"Failed to refresh context {context.id}: {str(e)}")
                raise ValueError(f"Context refresh failed: {str(e)}")
