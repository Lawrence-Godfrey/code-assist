"""
Command line interface for context management.

This module provides CLI commands for managing contexts, including adding, removing,
refreshing, and editing contexts from various sources like GitHub, Notion, etc.
"""

import os
from typing import Optional

from rich.console import Console
from rich.table import Table

from code_assistant.context.manager import (
    ContextManager,
)
from code_assistant.context.models import ContextMetadata, ContextSource, ContextType
from code_assistant.embedding.code_embedder import CodeEmbedder
from code_assistant.logging.logger import get_logger
from code_assistant.models.factory import ModelFactory

logger = get_logger(__name__)
console = Console()


class ContextCommands:
    """Commands for managing contexts."""

    async def add(
        self,
        type: str,
        source: str,
        title: str,
        description: str,
        source_url: str,
        database_url: str = "mongodb://localhost:27017/",
        **kwargs,
    ) -> None:
        """
        Add a new context.

        Examples:
            # Add GitHub repository
            code-assist context add \\
                --type="code" \\
                --source="github" \\
                --title="My Project" \\
                --description="Main project repo" \\
                --source-url="https://github.com/user/repo" \\
                --github-token="token"  # Optional

            # Add Notion database
            code-assist context add \\
                --type="document" \\
                --source="notion" \\
                --title="Project Docs" \\
                --description="Project documentation" \\
                --source-url="https://notion.so/..." \\
                --database-id="..." \\
                --notion-token="token"  # From env

        Args:
            type: Type of context ("code" or "document")
            source: Source type ("github", "notion", "confluence", "pdf")
            title: Human-readable title for the context
            description: Context description
            source_url: URL or path to source
            database_url: MongoDB connection URL
            **kwargs: Additional source-specific arguments
        """
        try:
            # Use environment variable if set
            database_url = os.getenv("MONGODB_URL") or database_url

            # Initialize manager
            manager = ContextManager(database_url)

            # Convert to enums
            ctx_type = ContextType(type.lower())
            ctx_source = ContextSource(source.lower())

            # Add source-specific metadata
            if ctx_source == ContextSource.GITHUB:
                kwargs["github_token"] = kwargs.get("github_token") or os.getenv(
                    "GITHUB_TOKEN"
                )

            elif ctx_source == ContextSource.NOTION:
                if not kwargs.get("database_id"):
                    raise ValueError("database_id is required for Notion sources")

            elif ctx_source == ContextSource.CONFLUENCE:
                if not kwargs.get("space_key"):
                    raise ValueError("space_key is required for Confluence sources")
                if not kwargs.get("base_url"):
                    raise ValueError("base_url is required for Confluence sources")

            # Add context
            context_id = await manager.add_context(
                type=ctx_type,
                source=ctx_source,
                title=title,
                description=description,
                source_url=source_url,
                **kwargs,
            )

            console.print(f"Added context: {title} ({context_id})")

        except Exception as e:
            logger.error(f"Failed to add context: {str(e)}")
            raise

    def remove(
        self,
        context_id: str,
        database_url: str = "mongodb://localhost:27017/",
    ) -> None:
        """
        Remove a context.

        Args:
            context_id: ID of context to remove
            database_url: MongoDB connection URL
        """
        try:
            database_url = os.getenv("MONGODB_URL") or database_url
            manager = ContextManager(database_url)

            # Remove context and all associated data
            manager.remove_context(context_id)
            console.print(f"Successfully removed context and all associated data")

        except Exception as e:
            logger.error(f"Failed to remove context: {str(e)}")
            raise

    def list(
        self,
        database_url: str = "mongodb://localhost:27017/",
    ) -> None:
        """
        List all available contexts.

        Args:
            database_url: MongoDB connection URL
        """
        try:
            database_url = os.getenv("MONGODB_URL") or database_url
            manager = ContextManager(database_url)

            contexts = manager.registry.list_contexts()

            if not contexts:
                console.print("No contexts found")
                return

            # Create table
            table = Table(show_header=True, header_style="bold")
            table.add_column("ID")
            table.add_column("Title")
            table.add_column("Type")
            table.add_column("Source")
            table.add_column("Description")
            table.add_column("Last Updated")

            for ctx in contexts:
                table.add_row(
                    ctx.id,
                    ctx.title,
                    ctx.type.value,
                    ctx.source.value,
                    ctx.description,
                    ctx.last_updated.strftime("%Y-%m-%d %H:%M:%S"),
                )

            console.print(table)

        except Exception as e:
            logger.error(f"Failed to list contexts: {str(e)}")
            raise

    async def refresh(
        self,
        context_id: Optional[str] = None,
        all: bool = False,
        database_url: str = "mongodb://localhost:27017/",
        **kwargs,
    ) -> None:
        """
        Refresh one or all contexts.

        Args:
            context_id: ID of context to refresh (required if all=False)
            all: Whether to refresh all contexts
            database_url: MongoDB connection URL
        """
        try:
            database_url = os.getenv("MONGODB_URL") or database_url
            manager = ContextManager(database_url)

            if not all and not context_id:
                raise ValueError("Must specify either context_id or --all")

            if all:
                console.print("Refreshing all contexts...")
                if not kwargs.get("github_token") or os.getenv("GITHUB_TOKEN"):
                    raise ValueError("github_token is required for Github sources")
                if not kwargs.get("database_id"):
                    raise ValueError("database_id is required for Notion sources")
                if not kwargs.get("space_key"):
                    raise ValueError("space_key is required for Confluence sources")
                if not kwargs.get("base_url"):
                    raise ValueError("base_url is required for Confluence sources")
            else:
                if context := manager.registry.get_context(context_id):
                    console.print(f"Refreshing context: {context.title}")
                else:
                    raise ValueError(f"Context {context_id} not found")

                if context.source == ContextSource.GITHUB:
                    if not kwargs.get("github_token") or os.getenv("GITHUB_TOKEN"):
                        raise ValueError("github_token is required for Github sources")
                elif context.source == ContextSource.NOTION:
                    if not kwargs.get("database_id"):
                        raise ValueError("database_id is required for Notion sources")
                elif context.source == ContextSource.CONFLUENCE:
                    if not kwargs.get("space_key"):
                        raise ValueError("space_key is required for Confluence sources")
                    if not kwargs.get("base_url"):
                        raise ValueError("base_url is required for Confluence sources")

            await manager.refresh_context(context_id=context_id, all=all, **kwargs)
            console.print("Refresh completed successfully")

        except Exception as e:
            logger.error(f"Failed to refresh context(s): {str(e)}")
            raise

    def edit(
        self,
        context_id: str,
        database_url: str = "mongodb://localhost:27017/",
        **updates,
    ) -> None:
        """
        Edit context metadata.

        Args:
            context_id: ID of context to edit
            database_url: MongoDB connection URL
            **updates: Field updates as keyword arguments
        """
        try:
            database_url = os.getenv("MONGODB_URL") or database_url
            manager = ContextManager(database_url)

            if context := manager.registry.get_context(context_id):
                manager.registry.update_context(context_id, **updates)
                console.print(f"Updated context: {context.title}")
            else:
                raise ValueError(f"Context {context_id} not found")

        except Exception as e:
            logger.error(f"Failed to edit context: {str(e)}")
            raise

    def embed(
        self,
        context_id: Optional[str] = None,
        all: bool = False,
        database_url: str = "mongodb://localhost:27017/",
        model_name: str = ModelFactory.get_default_embedding_model(),
    ) -> None:
        """
        Generate embeddings for a context or all contexts.

        Args:
            context_id: ID of context to embed (required if all=False)
            all: Whether to embed all contexts
            database_url: MongoDB connection URL
            model_name: Name of the model to use for embeddings
        """
        try:
            database_url = os.getenv("MONGODB_URL") or database_url
            manager = ContextManager(database_url)

            if not all and not context_id:
                raise ValueError("Must specify either context_id or --all")

            # Create the embedding model
            embedding_model = ModelFactory.create(model_name)

            if all:
                contexts = manager.registry.list_contexts()
                console.print(
                    f"Generating embeddings for all contexts using {model_name}"
                )
            else:
                if context := manager.registry.get_context(context_id):
                    contexts = [context]
                    console.print(f"Generating embeddings for context: {context.title}")
                else:
                    raise ValueError(f"Context {context_id} not found")

            for context in contexts:
                try:
                    # Get appropriate store
                    store = manager.get_store(context.type, context.title)

                    # Create embedder
                    embedder = CodeEmbedder(embedding_model=embedding_model)

                    # Generate embeddings
                    logger.info(f"Generating embeddings for {context.title}...")
                    units_processed = embedder.embed_code_units(store)

                    # Refresh vector indexes
                    store.refresh_vector_indexes()

                    console.print(
                        f"Processed {units_processed} units in {context.title} "
                        f"with embedding dimension {embedding_model.embedding_dimension}"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to generate embeddings for context {context.id}: {str(e)}"
                    )
                    if not all:  # Raise error if processing single context
                        raise

            console.print("Embedding generation completed successfully")

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
