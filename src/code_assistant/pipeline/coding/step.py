"""
Pipeline step for code coding.

This module implements the coding step in the pipeline, responsible for
setting up the environment, applying code changes, and running tests.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

from code_assistant.feedback.manager import FeedbackManager
from code_assistant.feedback.mixins import FeedbackEnabled
from code_assistant.logging.logger import get_logger
from code_assistant.models.prompt import PromptModel
from code_assistant.pipeline.coding.checkpoints.manager import CheckpointManager
from code_assistant.pipeline.coding.checkpoints.models import Checkpoint, CheckpointStatus
from code_assistant.pipeline.coding.environment.base import ExecutionResult
from code_assistant.pipeline.coding.environment.factory import EnvironmentFactory
from code_assistant.pipeline.coding.models import ChangeType, CodeChange
from code_assistant.pipeline.step import PipelineStep

logger = get_logger(__name__)


class CodingStep(PipelineStep, FeedbackEnabled):
    """Pipeline step for code execution."""

    def __init__(
        self,
        prompt_model: PromptModel,
        feedback_manager: FeedbackManager,
        env_type: str = "docker",
        env_config: Optional[Dict] = None,
        max_retry_attempts: int = 3,
    ):
        """
        Initialize the execution step.

        Args:
            prompt_model: The prompt model for code generation
            feedback_manager: The feedback manager for user interaction
            env_type: Type of execution environment ("docker", "venv")
            env_config: Optional configuration for environment
            max_retry_attempts: Maximum number of retry attempts for errors
        """
        PipelineStep.__init__(self)
        FeedbackEnabled.__init__(self, feedback_manager)

        self._prompt_model = prompt_model
        self._environment = EnvironmentFactory.create_environment(env_type, env_config)
        self._checkpoint_manager = CheckpointManager()
        self._max_retry_attempts = max_retry_attempts
        self._current_attempt = 0
        self._branch_name = None

    def execute(self, context: Dict) -> None:
        """
        Execute the pipeline step.

        Args:
            context: Pipeline context containing the task information

        Raises:
            ValueError: If required context is missing
        """
        if "requirements_schema" not in context:
            raise ValueError("Requirements schema not found in context")

        repo_url = os.getenv("REPO_URL")
        if not repo_url:
            raise ValueError("REPO_URL environment variable is not set")

        # Generate branch name based on requirements
        self._branch_name = self._generate_branch_name(context["requirements_schema"])

        try:
            # Setup environment
            logger.info("Setting up execution environment")
            self._environment.setup(repo_url, self._branch_name)

            # Initialize git config
            self._initialize_git_config()

            # Generate and apply code changes
            changes = self._generate_changes(context)
            apply_result = self._apply_changes(changes)

            if not apply_result.success:
                raise RuntimeError(f"Failed to apply changes: {apply_result.error or apply_result.output}")

            # Create checkpoint for changes
            checkpoint = self._checkpoint_manager.create_checkpoint(
                self._environment,
                self._generate_commit_message(context, changes),
                changes
            )

            # Run tests
            test_result = self._run_tests(checkpoint)

            # Update context with results
            context["coding_results"] = {
                "success": test_result.success,
                "changes": [str(c) for c in changes],
                "branch_name": self._branch_name,
                "checkpoint_id": checkpoint.id,
                "commit_hash": checkpoint.commit_hash,
            }

            # Push changes
            self._push_changes()

            # Request feedback on implementation
            self._request_implementation_feedback(context, checkpoint)

        except Exception as e:
            # Handle the error
            logger.error(f"Execution step failed: {str(e)}")
            self._handle_execution_error(e, context)
            raise

        # Continue to next step if successful
        return self.execute_next(context)

    def _generate_changes(self, context: Dict) -> List[CodeChange]:
        """
        Generate code changes based on requirements.

        In the current implementation, this creates a simple test file.
        In future versions, this will use LLM to generate sophisticated changes.

        Args:
            context: Pipeline context with requirements schema

        Returns:
            List of code changes to apply
        """
        logger.info("Generating code changes")

        # For now, implement a basic file creation
        test_code = (
            "import unittest\n\n"
            "class TestBasicFunctionality(unittest.TestCase):\n"
            "    def test_addition(self):\n"
            "        self.assertEqual(2 + 2, 4)\n\n"
            "    def test_string_operation(self):\n"
            "        self.assertEqual('hello' + ' world', 'hello world')\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )

        # Implementation file
        impl_code = (
            "def add(a, b):\n"
            "    \"\"\"\n"
            "    Add two numbers and return the result.\n"
            "    \n"
            "    Args:\n"
            "        a: First number\n"
            "        b: Second number\n"
            "        \n"
            "    Returns:\n"
            "        Sum of the two numbers\n"
            "    \"\"\"\n"
            "    return a + b\n\n"
            "def concatenate(str1, str2):\n"
            "    \"\"\"\n"
            "    Concatenate two strings and return the result.\n"
            "    \n"
            "    Args:\n"
            "        str1: First string\n"
            "        str2: Second string\n"
            "        \n"
            "    Returns:\n"
            "        Concatenated string\n"
            "    \"\"\"\n"
            "    return str1 + str2\n"
        )

        changes = [
            # Test file
            CodeChange(
                type=ChangeType.CREATE,
                file_path="test_basic.py",
                content=test_code
            ),
            # Implementation file
            CodeChange(
                type=ChangeType.CREATE,
                file_path="implementation.py",
                content=impl_code
            )
        ]

        return changes

    def _apply_changes(self, changes: List[CodeChange]) -> ExecutionResult:
        """
        Apply code changes to the environment.

        Args:
            changes: List of code changes to apply

        Returns:
            Result of applying the changes
        """
        logger.info(f"Applying {len(changes)} code changes")

        # Stage files for commit
        result = self._environment.apply_changes(changes)

        if result.success:
            # Stage all changes
            stage_result = self._environment.execute_command("git add .")
            if stage_result.exit_code != 0:
                return ExecutionResult(
                    exit_code=stage_result.exit_code,
                    output=stage_result.output,
                    error=f"Failed to stage changes: {stage_result.error or stage_result.output}"
                )

        return result

    def _run_tests(self, checkpoint: Checkpoint) -> ExecutionResult:
        """
        Run tests for the current implementation.

        Args:
            checkpoint: The checkpoint to validate with tests

        Returns:
            Result of test execution
        """
        logger.info("Running tests")

        # Find all test files
        find_tests_result = self._environment.execute_command(
            "find . -name 'test_*.py' -type f -not -path '*/\\.*'"
        )

        if find_tests_result.exit_code != 0:
            # Fall back to our known test file
            test_files = ["test_basic.py"]
        else:
            test_files = find_tests_result.output.strip().split("\n")
            # Remove empty entries and relative path prefix
            test_files = [f.strip("./") for f in test_files if f.strip()]

        # Run tests and validate checkpoint
        self._checkpoint_manager.validate_checkpoint(
            checkpoint.id,
            self._environment,
            test_files
        )

        # Get updated checkpoint with test results
        updated_checkpoint = self._checkpoint_manager.get_checkpoint(checkpoint.id)

        # Convert to ExecutionResult for consistency
        return ExecutionResult(
            exit_code=updated_checkpoint.test_results.exit_code,
            output=updated_checkpoint.test_results.output,
            error=updated_checkpoint.test_results.error
        )

    def _push_changes(self) -> ExecutionResult:
        """
        Push changes to the remote repository.

        Returns:
            Result of the push operation
        """
        logger.info(f"Pushing changes to branch: {self._branch_name}")

        # Push to remote
        return self._environment.execute_command(f"git push -u origin {self._branch_name}")

    def _handle_execution_error(self, error: Exception, context: Dict) -> None:
        """
        Handle errors that occur during execution.

        Args:
            error: The exception that occurred
            context: Pipeline context
        """
        self._current_attempt += 1

        # Determine if we should retry
        if self._current_attempt <= self._max_retry_attempts:
            logger.info(f"Retrying execution (attempt {self._current_attempt} of {self._max_retry_attempts})")

            # Get the latest successful checkpoint if any
            latest_valid_checkpoint = None
            for checkpoint in self._checkpoint_manager.get_checkpoints_by_status(CheckpointStatus.VALIDATED):
                if not latest_valid_checkpoint or checkpoint.timestamp > latest_valid_checkpoint.timestamp:
                    latest_valid_checkpoint = checkpoint

            # Roll back if we have a valid checkpoint
            if latest_valid_checkpoint:
                self._checkpoint_manager.rollback_to_checkpoint(
                    latest_valid_checkpoint.id,
                    self._environment
                )

            # Request feedback on how to fix the error
            response = self.request_step_feedback(
                context="error_resolution",
                prompt=(
                    f"An error occurred during execution:\n\n"
                    f"{str(error)}\n\n"
                    "Would you like to try a different approach? (yes/no)"
                )
            )

            if response.lower().strip() in ("y", "yes"):
                # Try again with potentially different changes
                self.execute(context)
            else:
                # User doesn't want to retry, cleanup and continue
                self._cleanup()
        else:
            # Max attempts reached, cleanup and continue
            logger.warning(f"Max retry attempts ({self._max_retry_attempts}) reached, giving up")
            self._cleanup()

            # Update context with failure info
            context["coding_results"] = {
                "success": False,
                "error": str(error),
                "attempts": self._current_attempt,
            }

    def _initialize_git_config(self) -> None:
        """Initialize git configuration in the environment."""
        # Set up Git config for commits
        self._environment.execute_command(
            "git config --local user.email 'code-assistant@example.com'"
        )
        self._environment.execute_command(
            "git config --local user.name 'Code Assistant'"
        )

    def _cleanup(self) -> None:
        """Clean up resources after execution."""
        try:
            logger.info("Cleaning up execution environment")
            self._environment.cleanup()
        except Exception as e:
            logger.error(f"Environment cleanup failed: {str(e)}")

    def _request_implementation_feedback(self, context: Dict, checkpoint: Checkpoint) -> None:
        """
        Request feedback on the implementation.

        Args:
            context: Pipeline context
            checkpoint: The implementation checkpoint
        """
        # Get the updated checkpoint with test results
        checkpoint = self._checkpoint_manager.get_checkpoint(checkpoint.id)

        # Generate implementation summary
        summary = (
            f"Implementation complete on branch: {self._branch_name}\n"
            f"Commit: {checkpoint.commit_hash[:8]}\n"
            f"Test results: {'PASSED' if checkpoint.test_results.success else 'FAILED'}\n"
            f"Tests passed: {checkpoint.test_results.passed_count}\n"
            f"Tests failed: {checkpoint.test_results.failed_count}\n"
            f"Test errors: {checkpoint.test_results.error_count}\n\n"
            "Changes implemented:\n"
        )

        for i, change in enumerate(checkpoint.changes, 1):
            summary += f"{i}. {change}\n"

        # Request feedback
        response = self.request_step_feedback(
            context="implementation_review",
            prompt=(
                f"{summary}\n\n"
                "Are you satisfied with this implementation? (yes/no)"
            )
        )

        # Store feedback in context
        context["implementation_feedback"] = response

        # If user is not satisfied, we could implement additional changes here
        if response.lower().strip() not in ("y", "yes"):
            # For now, just log this. In the future, we could handle this more robustly
            logger.info("User is not satisfied with the implementation. Consider additional changes.")

    def _generate_branch_name(self, requirements_schema) -> str:
        """
        Generate a branch name based on the requirements schema.

        Args:
            requirements_schema: The requirements schema

        Returns:
            A branch name
        """
        # Convert task type and description to kebab case
        task_type = requirements_schema.task_type.value if requirements_schema.task_type else "task"

        # Create a prefix based on task type
        if task_type == "implementation":
            prefix = "feat"
        elif task_type == "investigation":
            prefix = "explore"
        elif task_type == "design_document":
            prefix = "design"
        else:
            prefix = "task"

        # Create a slug from the description
        import re
        description = requirements_schema.description[:30]  # Limit length
        slug = re.sub(r'[^a-zA-Z0-9]', '-', description.lower())
        slug = re.sub(r'-+', '-', slug).strip('-')  # Remove duplicate hyphens

        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%m%d%H%M")

        return f"{prefix}/{slug}-{timestamp}"

    def _generate_commit_message(self, context: Dict, changes: List[CodeChange]) -> str:
        """
        Generate a descriptive commit message.

        Args:
            context: Pipeline context
            changes: The code changes being committed

        Returns:
            A commit message
        """
        requirements = context.get("requirements_schema")

        if requirements and requirements.description:
            # Use the first line of the description
            first_line = requirements.description.split('\n')[0]
            message = f"{first_line[:50]}"

            if len(first_line) > 50:
                message += "..."
        else:
            # Fall back to generic message
            file_types = {change.type.value: [] for change in changes}
            for change in changes:
                file_types[change.type.value].append(change.file_path)

            message = "Implement changes: "
            details = []

            if ChangeType.CREATE.value in file_types:
                details.append(f"created {len(file_types[ChangeType.CREATE.value])} files")
            if ChangeType.MODIFY.value in file_types:
                details.append(f"modified {len(file_types[ChangeType.MODIFY.value])} files")
            if ChangeType.DELETE.value in file_types:
                details.append(f"deleted {len(file_types[ChangeType.DELETE.value])} files")

            message += ", ".join(details)

        return message