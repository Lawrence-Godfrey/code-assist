import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import docker
import git
from docker.models.containers import Container

from code_assistant.logging.logger import get_logger
from code_assistant.pipeline.coding.environment.base import (
    ExecutionEnvironment,
    ExecutionResult,
)
from code_assistant.pipeline.coding.environment.exceptions import (
    ChangeApplicationError,
    CommandExecutionError,
    EnvironmentCleanupError,
    EnvironmentError,
    EnvironmentSetupError,
    TestExecutionError,
)
from code_assistant.pipeline.coding.models import (
    ChangeResult,
    ChangeType,
    CodeChange,
    ModificationType,
)

logger = get_logger(__name__)

from code_assistant.pipeline.coding.environment.config import DockerConfig


class DockerEnvironment(ExecutionEnvironment):
    """Docker-based execution environment."""

    def __init__(self, config: Optional[DockerConfig] = None):
        """Initialize Docker environment."""
        self.config = config or DockerConfig()
        self.client = docker.from_env()
        self.container: Optional[Container] = None
        self.work_dir: Optional[Path] = None
        self.repo: Optional[git.Repo] = None

    async def setup(self, repo_url: str, branch: str = "main") -> None:
        """
        Set up Docker environment with repository.

        Args:
            repo_url: URL of Git repository
            branch: Branch to check out

        Raises:
            EnvironmentError: If setup fails
        """
        try:
            # Create temporary working directory
            self.work_dir = Path(tempfile.mkdtemp())

            # Clone repository
            logger.info(f"Cloning repository from {repo_url}")
            self.repo = git.Repo.clone_from(repo_url, self.work_dir, branch=branch)

            # Create and start container
            logger.info("Creating Docker container")
            self.container = self.client.containers.run(
                image=self.config.base_image,
                command="tail -f /dev/null",  # Keep container running
                volumes={
                    str(self.work_dir): {"bind": self.config.working_dir, "mode": "rw"}
                },
                working_dir=self.config.working_dir,
                mem_limit=self.config.memory_limit,
                nano_cpus=int(self.config.cpu_limit * 1e9),
                # Convert to nano CPUs
                detach=True,
                remove=True,
            )

            # Initial container setup
            logger.info("Setting up container environment")
            self._setup_container()

        except Exception as e:
            await self.cleanup()  # Clean up on failure
            raise EnvironmentSetupError(f"Environment setup failed: {str(e)}") from e

    def _setup_container(self) -> None:
        """Perform initial container setup."""
        # Install git
        self.execute_command("apt-get update && apt-get install -y git")

        # Install Python dependencies if requirements.txt exists
        if (self.work_dir / "requirements.txt").exists():
            self.execute_command("pip install -r requirements.txt")

    def execute_command(
        self,
        command: str,
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute a command in the container.

        Args:
            command: Command to execute
            workdir: Working directory for command
            env: Environment variables

        Returns:
            ExecutionResult with exit code and output

        Raises:
            EnvironmentError: If command execution fails
        """
        if not self.container:
            raise EnvironmentError("Container not initialized or has been cleaned up")

        try:
            exec_result = self.container.exec_run(
                ["/bin/bash", "-c", command],
                workdir=workdir or self.config.working_dir,
                environment=env,
            )

            return ExecutionResult(
                exit_code=exec_result.exit_code,
                output=exec_result.output.decode("utf-8"),
            )

        except Exception as e:
            raise CommandExecutionError(
                command=command, exit_code=-1, output="", error=str(e)
            ) from e

    async def cleanup(self) -> None:
        """Clean up Docker environment."""
        try:
            # Stop and remove container
            if self.container:
                try:
                    self.container.stop(timeout=5)
                except Exception as e:
                    logger.warning(f"Error stopping container: {str(e)}")

            # Remove working directory
            if self.work_dir:
                try:
                    import shutil

                    def handle_error(action, name, exc):
                        # Skip permission errors for __pycache__
                        if isinstance(exc, PermissionError) and "__pycache__" in str(
                            name
                        ):
                            return
                        raise exc

                    shutil.rmtree(self.work_dir, onerror=handle_error)
                except Exception as e:
                    logger.warning(f"Error removing working directory: {str(e)}")

        except Exception as e:
            logger.error(f"Environment cleanup failed: {str(e)}")
            raise EnvironmentCleanupError(
                f"Environment cleanup failed: {str(e)}"
            ) from e
        finally:
            self.container = None
            self.work_dir = None
            self.repo = None

    def _apply_single_change(self, change: CodeChange) -> ChangeResult:
        """
        Apply a single code change in the Docker container.

        Args:
            change: The change to apply

        Returns:
            ChangeResult indicating success/failure
        """
        try:
            file_path = Path(self.config.working_dir) / change.file_path

            if change.type == ChangeType.CREATE:
                if not change.content:
                    return ChangeResult(
                        False,
                        str(change.file_path),
                        "No content provided for file creation",
                    )

                # Create parent directories
                self.execute_command(f"mkdir -p {file_path.parent}")

                # Write file content using echo and redirection
                # We use printf to handle special characters and multiline content properly
                result = self.execute_command(
                    f"printf '%s' '{change.content}' > {file_path}"
                )
                if result.exit_code != 0:
                    return ChangeResult(
                        False,
                        str(change.file_path),
                        f"Failed to create file: {result.error or result.output}",
                    )

            elif change.type == ChangeType.DELETE:
                result = self.execute_command(f"rm -f {file_path}")
                if result.exit_code != 0:
                    return ChangeResult(
                        False,
                        str(change.file_path),
                        f"Failed to delete file: {result.error or result.output}",
                    )

            elif change.type == ChangeType.MODIFY:
                if not change.modifications:
                    return ChangeResult(
                        False, str(change.file_path), "No modifications provided"
                    )

                # Check if file exists
                result = self.execute_command(f"test -f {file_path}")
                if result.exit_code != 0:
                    return ChangeResult(
                        False, str(change.file_path), "File does not exist"
                    )

                # Read existing content
                result = self.execute_command(f"cat {file_path}")
                if result.exit_code != 0:
                    return ChangeResult(
                        False,
                        str(change.file_path),
                        f"Failed to read file: {result.error or result.output}",
                    )

                lines = result.output.splitlines(keepends=True)

                # Apply modifications in reverse order (to maintain line numbers)
                for mod in sorted(
                    change.modifications, key=lambda m: m.start_line or 0, reverse=True
                ):
                    if mod.type == ModificationType.INSERT:
                        if mod.start_line is None:
                            return ChangeResult(
                                False,
                                str(change.file_path),
                                "Line number required for insert",
                            )
                        lines.insert(mod.start_line - 1, mod.content + "\n")

                    elif mod.type == ModificationType.REPLACE:
                        if mod.start_line is None or mod.end_line is None:
                            return ChangeResult(
                                False,
                                str(change.file_path),
                                "Line numbers required for replace",
                            )
                        lines[mod.start_line - 1 : mod.end_line] = [mod.content + "\n"]

                    elif mod.type == ModificationType.DELETE:
                        if mod.start_line is None or mod.end_line is None:
                            return ChangeResult(
                                False,
                                str(change.file_path),
                                "Line numbers required for delete",
                            )
                        del lines[mod.start_line - 1 : mod.end_line]

                # Write back modified content
                content = "".join(lines)
                result = self.execute_command(f"printf '%s' '{content}' > {file_path}")
                if result.exit_code != 0:
                    return ChangeResult(
                        False,
                        str(change.file_path),
                        f"Failed to write modifications: {result.error or result.output}",
                    )

            return ChangeResult(True, str(change.file_path))

        except Exception as e:
            return ChangeResult(False, str(change.file_path), str(e))

    async def apply_changes(self, changes: List[CodeChange]) -> ExecutionResult:
        """
        Apply code changes in the environment.

        Args:
            changes: List of code changes to apply

        Returns:
            ExecutionResult indicating success/failure

        Raises:
            ChangeApplicationError: If changes cannot be applied
        """
        try:
            results = []
            failed_changes = []

            # Apply each change
            for change in changes:
                result = self._apply_single_change(change)
                results.append(result)
                if not result.success:
                    failed_changes.append(result)

            if failed_changes:
                error_msg = "\n".join(
                    f"{result.file_path}: {result.error}" for result in failed_changes
                )
                raise ChangeApplicationError(
                    error=f"Failed to apply changes:\n{error_msg}", changes=changes
                )

            # Run any necessary build/compile steps
            result = self.execute_command(f"test -f {self.config.working_dir}/setup.py")
            if result.exit_code == 0:
                result = self.execute_command("python setup.py develop")
                if result.exit_code != 0:
                    raise ChangeApplicationError(
                        error=result.error or result.output, changes=changes
                    )

            return ExecutionResult(
                exit_code=0, output=f"Successfully applied {len(changes)} changes"
            )

        except Exception as e:
            raise ChangeApplicationError(error=str(e), changes=changes) from e

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
        try:
            # Install test dependencies
            self.execute_command("pip install pytest pytest-cov")

            # Run tests
            test_command = (
                f"pytest {' '.join(test_files)} " "--cov=. --cov-report=term-missing"
            )

            return self.execute_command(test_command)

        except Exception as e:
            raise TestExecutionError(
                test_files=test_files, output="", error=str(e)
            ) from e
