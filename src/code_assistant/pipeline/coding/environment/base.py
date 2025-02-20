"""
Base classes and interfaces for execution environments.

This module defines the fundamental interfaces and data structures for
execution environments, ensuring consistent patterns across different
implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ExecutionResult:
    """Result of code execution in environment."""
    exit_code: int
    output: str
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether the execution was successful."""
        return self.exit_code == 0


class ExecutionEnvironment(ABC):
    """
    Abstract base class for execution environments.

    This class defines the interface that all execution environments must
    implement, ensuring consistent behavior across different environment
    types (Docker, virtual env, etc).
    """

    @abstractmethod
    async def setup(self, repo_url: str, branch: str = "main") -> None:
        """
        Set up the execution environment.

        Args:
            repo_url: URL of Git repository to clone
            branch: Branch to check out

        Raises:
            EnvironmentError: If setup fails
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up the execution environment.

        This should release all resources and restore the system to its
        original state.

        Raises:
            EnvironmentError: If cleanup fails
        """
        pass

    @abstractmethod
    def execute_command(
            self,
            command: str,
            workdir: Optional[str] = None,
            env: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """
        Execute a command in the environment.

        Args:
            command: Command to execute
            workdir: Working directory for command execution
            env: Environment variables to set

        Returns:
            ExecutionResult containing exit code and output

        Raises:
            EnvironmentError: If command execution fails
        """
        pass

    @abstractmethod
    async def apply_changes(self, changes: List[str]) -> ExecutionResult:
        """
        Apply code changes in the environment.

        Args:
            changes: List of code changes to apply

        Returns:
            ExecutionResult indicating success/failure

        Raises:
            EnvironmentError: If changes cannot be applied
        """
        pass

    @abstractmethod
    async def run_tests(self, test_files: List[str]) -> ExecutionResult:
        """
        Run tests in the environment.

        Args:
            test_files: List of test files to run

        Returns:
            ExecutionResult with test output

        Raises:
            EnvironmentError: If tests fail to run
        """
        pass