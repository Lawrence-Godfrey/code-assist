import asyncio
import os
from typing import Optional

from code_assistant.data_extraction.extractors.confluence_document_extractor import ConfluenceDocumentExtractor, SpaceExistsError
from code_assistant.data_extraction.extractors.github_code_extractor import (
    CodebaseExistsError,
    GitHubCodeExtractor,
)
from code_assistant.logging.logger import get_logger
from code_assistant.storage.stores.code import MongoDBCodeStore
from code_assistant.storage.stores.document import MongoDBDocumentStore

logger = get_logger(__name__)


class ExtractCommands:
    """Commands for extracting code from repositories."""

    def github(
        self,
        repo_url: str,
        database_url: str = "mongodb://localhost:27017/",
        max_files: Optional[int] = None,
        github_token: Optional[str] = None,
        repo_download_dir: Optional[str] = os.path.expanduser(
            "~/code_assist/github_repos"
        ),
        overwrite: bool = False,
    ) -> None:
        """Extract code units from a GitHub repository."""
        extractor = GitHubCodeExtractor(
            repo_download_dir=repo_download_dir, github_token=github_token
        )

        repo = extractor.clone_repository(repo_url=repo_url)

        database_url = os.getenv("MONGODB_URL") or database_url

        code_store = MongoDBCodeStore(
            codebase=repo.name, connection_string=database_url
        )

        try:
            extractor.process_repository(
                repo=repo,
                code_store=code_store,
                max_files=max_files,
                overwrite=overwrite,
            )
        except CodebaseExistsError:
            logger.error(
                f"Codebase {repo.full_name} already exists in the storage. Use --overwrite to replace."
            )

    def confluence(
            self,
            space_key: str,
            base_url: str,
            database_url: str = "mongodb://localhost:27017/",
            limit: Optional[int] = None,
            overwrite: bool = False,
    ) -> None:
        """
        Extract documents from a Confluence space.

        Args:
            space_key: Key of the Confluence space to extract from
            base_url: Base URL of the Confluence instance
            database_url: MongoDB connection URL
            limit: Maximum number of pages to process
            overwrite: Whether to overwrite existing space content
        """
        try:
            # Initialize extractor
            extractor = ConfluenceDocumentExtractor(base_url=base_url)

            # Setup document store
            database_url = os.getenv("MONGODB_URL") or database_url
            doc_store = MongoDBDocumentStore(
                space_key=space_key,
                connection_string=database_url
            )

            # Run extraction (needs to be run in asyncio event loop)
            asyncio.run(
                extractor.extract_documents(
                    space_key=space_key,
                    doc_store=doc_store,
                    limit=limit,
                    overwrite=overwrite,
                )
            )

        except SpaceExistsError:
            logger.error(
                f"Space {space_key} already exists in storage. Use --overwrite to replace."
            )
        except ValueError as e:
            logger.error(str(e))
        except Exception as e:
            logger.error(f"Error extracting from Confluence: {str(e)}")