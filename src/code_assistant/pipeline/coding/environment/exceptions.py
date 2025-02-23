"""
Custom exceptions for execution environments.

This module defines the exception hierarchy used throughout the execution
environment system to handle error cases and environment failures.
"""

from code_assistant.pipeline.coding.models import CodeChange


class EnvironmentError(Exception):
    """Base class for all environment-related errors."""

    pass


class EnvironmentSetupError(EnvironmentError):
    """Raised when environment setup fails."""

    pass


class EnvironmentCleanupError(EnvironmentError):
    """Raised when environment cleanup fails."""

    pass


class CommandExecutionError(EnvironmentError):
    """Raised when command execution fails."""

    def __init__(self, command: str, exit_code: int, output: str, error: str = None):
        self.command = command
        self.exit_code = exit_code
        self.output = output
        self.error = error
        super().__init__(
            f"Command '{command}' failed with exit code {exit_code}.\n"
            f"Output: {output}\n"
            f"Error: {error or 'None'}"
        )


class TestExecutionError(EnvironmentError):
    """Raised when test execution fails."""

    def __init__(self, test_files: list[str], output: str, error: str = None):
        self.test_files = test_files
        self.output = output
        self.error = error
        super().__init__(
            f"Test execution failed for files: {', '.join(test_files)}.\n"
            f"Output: {output}\n"
            f"Error: {error or 'None'}"
        )


class ChangeApplicationError(EnvironmentError):
    """Raised when code changes cannot be applied."""

    def __init__(self, error: str, changes: list[CodeChange] = None):
        self.error = error
        self.changes = changes
        super().__init__(f"Failed to apply code changes: {error}")
