"""
Checkpoint management system for tracking code changes.

This module provides functionality for creating, tracking, and managing code
checkpoints throughout the development process, enabling rollback capabilities
and progress tracking.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from code_assistant.logging.logger import get_logger
from code_assistant.pipeline.coding.checkpoints.models import (
    Checkpoint,
    CheckpointStatus,
    TestResult,
)
from code_assistant.pipeline.coding.environment.base import ExecutionEnvironment
from code_assistant.pipeline.coding.models import CodeChange

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages development checkpoints for the coding pipeline.

    Provides functionality for creating, retrieving, and rolling back to checkpoints,
    as well as storing checkpoint metadata and test results.
    """

    def __init__(self):
        """Initialize the checkpoint manager."""
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._current_branch: Optional[str] = None

    async def create_checkpoint(
        self, environment: ExecutionEnvironment, message: str, changes: List[CodeChange]
    ) -> Checkpoint:
        """
        Create a new checkpoint for the current code state.

        Args:
            environment: The execution environment to create the checkpoint in
            message: Commit message for the checkpoint
            changes: List of code changes included in this checkpoint

        Returns:
            Newly created Checkpoint object

        Raises:
            EnvironmentError: If the checkpoint creation fails
        """
        logger.info(f"Creating checkpoint: {message}")

        # Generate unique ID for this checkpoint
        checkpoint_id = str(uuid.uuid4())

        # Create the checkpoint in the environment (Git commit)
        result = environment.execute_command(f'git commit -m "{message}"')
        if result.exit_code != 0:
            logger.error(
                f"Failed to create checkpoint: {result.error or result.output}"
            )
            raise RuntimeError(
                f"Failed to create checkpoint: {result.error or result.output}"
            )

        # Get the commit hash from the result
        # Extract hash from output like: [main 1a2b3c4] Commit message
        commit_hash = ""
        if result.output:
            import re

            match = re.search(r"\[.*\s([0-9a-f]+)]", result.output)
            if match:
                commit_hash = match.group(1)

        if not commit_hash:
            # If we couldn't extract it, try getting it from git
            rev_parse_result = environment.execute_command("git rev-parse HEAD")
            if rev_parse_result.exit_code == 0:
                commit_hash = rev_parse_result.output.strip()
            else:
                commit_hash = "unknown"  # Fallback

        # Get current branch name
        branch_result = environment.execute_command("git rev-parse --abbrev-ref HEAD")
        branch_name = (
            branch_result.output.strip() if branch_result.exit_code == 0 else "unknown"
        )
        self._current_branch = branch_name

        # Create checkpoint object
        checkpoint = Checkpoint(
            id=checkpoint_id,
            commit_hash=commit_hash,
            message=message,
            branch_name=branch_name,
            timestamp=datetime.now(),
            status=CheckpointStatus.CREATED,
            changes=[str(change) for change in changes],
        )

        # Store the checkpoint
        self._checkpoints[checkpoint_id] = checkpoint
        logger.info(
            f"Created checkpoint {checkpoint_id} with commit {commit_hash[:8]} on branch {branch_name}"
        )

        return checkpoint

    async def validate_checkpoint(
        self,
        checkpoint_id: str,
        environment: ExecutionEnvironment,
        test_files: Optional[List[str]] = None,
    ) -> Checkpoint:
        """
        Run tests to validate a checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to validate
            environment: Execution environment to run tests in
            test_files: Optional list of specific test files to run

        Returns:
            Updated checkpoint with test results

        Raises:
            ValueError: If checkpoint not found
            EnvironmentError: If test execution fails
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        logger.info(f"Validating checkpoint {checkpoint_id}")

        # Run tests
        test_result = await environment.run_tests(test_files or [])

        # Extract test statistics from output
        passed_count, failed_count, error_count = self._parse_test_statistics(
            test_result.output
        )

        # Update the test results
        checkpoint.test_results = TestResult(
            success=test_result.exit_code == 0,
            exit_code=test_result.exit_code,
            output=test_result.output,
            passed_count=passed_count,
            failed_count=failed_count,
            error_count=error_count,
            error=test_result.error,
        )

        # Update status based on test results
        checkpoint.status = (
            CheckpointStatus.VALIDATED
            if test_result.exit_code == 0
            else CheckpointStatus.FAILED
        )

        logger.info(
            f"Checkpoint {checkpoint_id} validation "
            f"{'succeeded' if checkpoint.test_results.success else 'failed'}"
        )

        return checkpoint

    async def rollback_to_checkpoint(
        self,
        checkpoint_id: str,
        environment: ExecutionEnvironment,
    ) -> Checkpoint:
        """
        Roll back to a specific checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to roll back to
            environment: Execution environment to perform rollback in

        Returns:
            The checkpoint that was rolled back to

        Raises:
            ValueError: If checkpoint not found
            EnvironmentError: If rollback fails
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        logger.info(
            f"Rolling back to checkpoint {checkpoint_id} ({checkpoint.commit_hash[:8]})"
        )

        # Perform the rollback
        result = environment.execute_command(
            f"git reset --hard {checkpoint.commit_hash}"
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to roll back: {result.error or result.output}")

        # Update checkpoint status
        for cp in self._checkpoints.values():
            if cp.timestamp > checkpoint.timestamp:
                cp.status = CheckpointStatus.ROLLED_BACK

        logger.info(f"Successfully rolled back to checkpoint {checkpoint_id}")

        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the most recent checkpoint."""
        if not self._checkpoints:
            return None

        return max(self._checkpoints.values(), key=lambda cp: cp.timestamp)

    def get_checkpoints_by_status(self, status: CheckpointStatus) -> List[Checkpoint]:
        """Get all checkpoints with a specific status."""
        return [cp for cp in self._checkpoints.values() if cp.status == status]

    def _parse_test_statistics(self, test_output: str) -> tuple[int, int, int]:
        """
        Parse test statistics from pytest output.

        Returns:
            Tuple of (passed_count, failed_count, error_count)
        """
        # Default values
        passed_count, failed_count, error_count = 0, 0, 0

        # Look for pytest summary line like:
        # == 5 passed, 2 failed, 1 error in 0.71s ==
        import re

        summary_match = re.search(
            r"=+ (\d+ passed)?,? ?(\d+ failed)?,? ?(\d+ error)?", test_output
        )

        if summary_match:
            # Extract numbers from each group
            for group in summary_match.groups():
                if group:
                    num = int(re.search(r"(\d+)", group).group(1))
                    if "passed" in group:
                        passed_count = num
                    elif "failed" in group:
                        failed_count = num
                    elif "error" in group:
                        error_count = num

        return passed_count, failed_count, error_count
