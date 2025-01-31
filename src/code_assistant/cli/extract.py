import os
from typing import Optional

from code_assistant.data_extraction.extractors.github_code_extractor import (
    CodebaseExistsError,
    GitHubCodeExtractor,
)
from code_assistant.logging.logger import get_logger
from code_assistant.storage.stores import MongoDBCodeStore

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
