from pathlib import Path
from typing import Optional
import os

from data_extraction.extractors.github_code_extractor import GitHubCodeExtractor


class ExtractCommands:
    """Commands for extracting code from repositories."""

    def github(
        self,
        repo_url: str,
        output_path: Optional[str] = "code_units.json",
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

        codebase = extractor.process_repository(
            repo_url=repo_url,
            output_path=Path(output_path).parent,
            max_files=max_files,
            cleanup=True,
        )

        codebase.to_json(Path(output_path))
