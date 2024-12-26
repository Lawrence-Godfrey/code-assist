import ast
import logging
import os
import shutil
from os.path import expanduser
from pathlib import Path
from typing import List, Optional

import astor
import git
from git import Repo

from code_assistant.storage.code_store import (
    Class,
    CodebaseSnapshot,
    File,
    Function,
    Method,
)


class GitHubCodeExtractor:
    """
    Extract code units from a GitHub repository, including private repositories.
    """

    def __init__(
        self,
        repo_download_dir=os.path.expanduser("~/code_assist/github_repos"),
        github_token: Optional[str] = None,
    ):
        """
        Initialize the GitHub code extractor with optional private repo support.

        Args:
            repo_download_dir (str): Directory to clone repositories into
            github_token (str, optional): GitHub Personal Access Token for private repo access
        """
        # Ensure temp directory exists
        self.repo_download_dir = repo_download_dir
        os.makedirs(repo_download_dir, exist_ok=True)
        self.repo_path: Optional[str] = None
        self.repo_name: Optional[str] = None

        self.github_token = github_token

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _get_authenticated_repo_url(self, repo_url: str) -> str:
        """
        Convert a GitHub repository URL to an authenticated version.

        Args:
            repo_url (str): Original repository URL

        Returns:
            str: Authenticated repository URL
        """
        if not self.github_token:
            return repo_url

        # Extract repository path (username/repo)
        repo_path = "/".join(repo_url.split("/")[-2:])

        # Construct authenticated URL
        return f"https://{self.github_token}@github.com/{repo_path}"

    def clone_repository(self, repo_url: str) -> str:
        """
        Clone a GitHub repository to a local directory, supporting private repos.

        Args:
            repo_url (str): GitHub repository URL

        Returns:
            str: Path to the cloned repository
        """
        try:
            # Generate a safe directory name from the repo URL
            self.repo_name = repo_url.split("/")[-1].replace(".git", "")
            self.repo_path = os.path.join(self.repo_download_dir, self.repo_name)

            if os.path.exists(self.repo_path):
                shutil.rmtree(self.repo_path)

            auth_url = self._get_authenticated_repo_url(repo_url)

            self.logger.info(f"Cloning repository: {repo_url}")

            # Use custom clone method to handle authentication
            Repo.clone_from(
                auth_url,
                self.repo_path,
                env={
                    # Disable Git's credential helper to prevent hanging
                    "GIT_TERMINAL_PROMPT": "0"
                },
            )

            return self.repo_path

        except git.GitCommandError as e:
            # Provide more detailed error handling
            if "Authentication failed" in str(e):
                self.logger.error(
                    "Authentication failed. Please check your GitHub token and repository access."
                )
            elif "repository not found" in str(e).lower():
                self.logger.error(
                    "Repository not found. Verify the repository URL and your access permissions."
                )
            else:
                self.logger.error(f"Repository cloning failed: {e}")
            raise

    def find_python_files(self) -> List[str]:
        """
        Recursively find all Python files in the repository.

        Returns:
            List of paths to Python files
        """
        python_files = []

        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    python_files.append(full_path)

        self.logger.info(f"Found {len(python_files)} Python files")
        return python_files

    def extract_code_units(self, file_path: str) -> File:
        """
        Extract methods, classes, and functions from a Python file.

        Args:
            file_path (str): Path to a Python source file

        Returns:
            A File object containing code units
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            tree = ast.parse(file_content)

            # Get the file path relative to the repository root.
            file_path_relative = os.path.relpath(file_path, self.repo_path)

            file = File(
                name=os.path.basename(file_path),
                filepath=Path(file_path_relative),
                source_code=file_content,
            )

            for node in ast.iter_child_nodes(tree):
                # Extract classes
                if isinstance(node, ast.ClassDef):
                    cls = Class(
                        name=node.name,
                        source_code=astor.to_source(node),
                        docstring=ast.get_docstring(node) or "",
                    )
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            cls.add_method(
                                Method(
                                    name=method.name,
                                    source_code=astor.to_source(method),
                                    docstring=ast.get_docstring(method) or "",
                                )
                            )

                    file.add_code_unit(cls)

                # Extract top-level functions
                elif isinstance(node, ast.FunctionDef):
                    file.add_code_unit(
                        Function(
                            name=node.name,
                            source_code=astor.to_source(node),
                            docstring=ast.get_docstring(node) or "",
                        )
                    )

            return file

        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            raise

    def process_repository(
        self,
        repo_url: str,
        output_path: Optional[Path] = Path(
            expanduser("~/code_assist/extracted_code_units")
        ),
        max_files: Optional[int] = None,
        cleanup: bool = True,
    ) -> CodebaseSnapshot:
        """
        Process an entire GitHub repository and extract code units.

        Args:
            repo_url (str): GitHub repository URL
            output_path (str, optional): Path to save extracted code units
            max_files (int, optional): Limit the number of files to process
            cleanup (bool): Whether to remove the cloned repository after processing

        Returns:
            A CodebaseSnapshot containing all extracted code units
        """
        repo_path = self.clone_repository(repo_url)

        python_files = self.find_python_files()

        # Optional: Limit number of files
        if max_files:
            python_files = python_files[:max_files]

        snapshot = CodebaseSnapshot()
        # Extract code units from all files
        for file_path in python_files:
            snapshot.add_file(self.extract_code_units(file_path))

        # Optionally save to JSON
        if output_path:
            output_file = os.path.join(output_path, f"{self.repo_name}_code_units.json")
            os.makedirs(output_path, exist_ok=True)
            snapshot.to_json(Path(output_file))

        # Clean up temporary repository
        if cleanup:
            shutil.rmtree(repo_path)

        self.logger.info(f"Extracted {len(snapshot)} code units")
        return snapshot
