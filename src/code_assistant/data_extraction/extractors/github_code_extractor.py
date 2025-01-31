import ast
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import astor
import git
from git import Repo as GitRepo

from code_assistant.logging.logger import get_logger
from code_assistant.storage.codebase import Class, File, Function, Method
from code_assistant.storage.stores import CodeStore

logger = get_logger(__name__)


@dataclass
class Repo:
    """Represents a repository with its metadata and local clone."""

    url: str
    name: str
    owner: str
    local_path: Path
    default_branch: str
    description: Optional[str] = None

    @classmethod
    def from_url(cls, url: str, local_path: Path) -> "Repo":
        """
        Create a Repo instance from a GitHub URL.

        Args:
            url: GitHub repository URL
            local_path: Path where repository is/will be cloned

        Returns:
            Repo instance with parsed metadata

        Examples:
            >>> repo = Repo.from_url("https://github.com/owner/repo", Path("/tmp/repos"))
            >>> repo.name
            'repo'
            >>> repo.owner
            'owner'
        """
        # Parse URL to extract owner and repo name
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub URL format: {url}")

        owner, name = path_parts[:2]

        # Remove .git suffix if present
        name = re.sub(r"\.git$", "", name)

        return cls(
            url=url,
            name=name,
            owner=owner,
            local_path=local_path / name,
            default_branch="main",  # Will be updated after cloning
        )

    @property
    def full_name(self) -> str:
        """Get the full repository name (owner/repo)."""
        return f"{self.owner}/{self.name}"

    def is_cloned(self) -> bool:
        """Check if repository is already cloned locally."""
        return self.local_path.exists() and self.local_path.is_dir()


class CodebaseExistsError(Exception):
    """Raised when a codebase already exists in the storage."""

    pass


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
        self.repo_download_dir = Path(repo_download_dir)
        self.repo_download_dir.mkdir(parents=True, exist_ok=True)

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

    def clone_repository(self, repo_url: str) -> Repo:
        """
        Clone a GitHub repository to a local directory, supporting private repos.

        Args:
            repo_url (str): GitHub repository URL

        Returns:
            Repo: Repository object with local path and metadata
        """
        try:
            # Create Repo object from URL
            repo = Repo.from_url(repo_url, self.repo_download_dir)

            # Clear existing clone if present
            if repo.is_cloned():
                shutil.rmtree(repo.local_path)

            auth_url = self._get_authenticated_repo_url(repo_url)

            self.logger.info(f"Cloning repository: {repo_url}")

            # Use custom clone method to handle authentication
            GitRepo.clone_from(
                auth_url,
                repo.local_path,
                env={
                    # Disable Git's credential helper to prevent hanging
                    "GIT_TERMINAL_PROMPT": "0"
                },
            )

            return repo

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

    def find_python_files(self, repo: Repo) -> List[Path]:
        """
        Recursively find all Python files in the repository.

        Args:
            repo (Repo): Repository object

        Returns:
            List of paths to Python files
        """
        python_files = []

        for root, _, files in os.walk(repo.local_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    python_files.append(Path(full_path))

        self.logger.info(f"Found {len(python_files)} Python files")
        return python_files

    def extract_code_units(self, file_path: Path, repo: Repo) -> File:
        """
        Extract methods, classes, and functions from a Python file.

        Args:
            file_path: Path to a Python source file
            repo: Repository object

        Returns:
            A File object containing code units
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            tree = ast.parse(file_content)

            # Get the file path relative to the repository root.
            file_path_relative = os.path.relpath(file_path, repo.local_path)

            file = File(
                name=os.path.basename(file_path),
                filepath=Path(file_path_relative),
                codebase=repo.name,
                source_code=file_content,
            )

            for node in ast.iter_child_nodes(tree):
                # Extract classes
                if isinstance(node, ast.ClassDef):
                    cls = Class(
                        name=node.name,
                        source_code=astor.to_source(node),
                        docstring=ast.get_docstring(node) or "",
                        codebase=repo.name,
                        filepath=file_path_relative,
                    )
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            cls.add_method(
                                Method(
                                    name=method.name,
                                    source_code=astor.to_source(method),
                                    docstring=ast.get_docstring(method) or "",
                                    codebase=repo.name,
                                    filepath=file_path_relative,
                                    classname=cls.name,
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
                            codebase=repo.name,
                            filepath=file_path_relative,
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
        repo: Repo,
        code_store: CodeStore,
        max_files: Optional[int] = None,
        overwrite: bool = False,
    ):
        """
        Process an entire GitHub repository and extract code units.

        Args:
            repo: The repository to process.
            code_store: Storage object to save the extracted code units
            max_files: Limit the number of files to process
            overwrite: Overwrite existing codebase if it exists
        """

        python_files = self.find_python_files(repo)

        # Optional: Limit number of files
        if max_files:
            python_files = python_files[:max_files]

        if overwrite:
            logger.info(f"Overwriting codebase {repo.name}")
            code_store.delete_codebase()
        else:
            if code_store.codebase_exists():
                raise CodebaseExistsError(f"Codebase {repo.name} already exists.")

        # Extract code units from all files
        for file_path in python_files:
            code_store.save_unit(self.extract_code_units(file_path, repo))

        self.logger.info(f"Extracted {len(code_store)} code units")
