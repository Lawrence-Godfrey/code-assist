import os
from pathlib import Path
from typing import Optional

from code_assistant.data_extraction.extractors.github_code_extractor import (
    GitHubCodeExtractor,
)
from code_assistant.storage.stores import MongoDBCodeStore, JSONCodeStore


class ExtractCommands:
    """Commands for extracting code from repositories."""

    def github(
        self,
        repo_url: str,
        output_path: Optional[str] = None,
        database_url: str = "mongodb://localhost:27017/",
        max_files: Optional[int] = None,
        github_token: Optional[str] = None,
        repo_download_dir: Optional[str] = os.path.expanduser(
            "~/code_assist/github_repos"
        ),
    ) -> None:
        """Extract code units from a GitHub repository."""
        extractor = GitHubCodeExtractor(
            repo_download_dir=repo_download_dir, github_token=github_token
        )

        if output_path:
            code_store = JSONCodeStore(Path(output_path))
        elif database_url:
            code_store = MongoDBCodeStore(database_url)
        else:
            raise ValueError("Either output_path or database_url must be provided.")

        extractor.process_repository(
            repo_url=repo_url,
            code_store=code_store,
            max_files=max_files,
        )
