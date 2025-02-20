from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import docker
import git
from docker.models.containers import Container

from code_assistant.pipeline.coding.environment.base import \
    ExecutionEnvironment, ExecutionResult
from code_assistant.pipeline.coding.environment.exceptions import EnvironmentError
from code_assistant.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DockerConfig:
    """Configuration for Docker environment."""
    base_image: str = "python:3.8"
    memory_limit: str = "2g"
    cpu_limit: int = 2
    timeout: int = 300
    working_dir: str = "/workspace"


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
            self.repo = git.Repo.clone_from(
                repo_url,
                self.work_dir,
                branch=branch
            )

            # Create and start container
            logger.info("Creating Docker container")
            self.container = self.client.containers.run(
                image=self.config.base_image,
                command="tail -f /dev/null",  # Keep container running
                volumes={
                    str(self.work_dir): {
                        'bind': self.config.working_dir,
                        'mode': 'rw'
                    }
                },
                working_dir=self.config.working_dir,
                mem_limit=self.config.memory_limit,
                nano_cpus=int(self.config.cpu_limit * 1e9),
                # Convert to nano CPUs
                detach=True,
                remove=True
            )

            # Initial container setup
            logger.info("Setting up container environment")
            self._setup_container()

        except Exception as e:
            await self.cleanup()  # Clean up on failure
            raise EnvironmentError(f"Environment setup failed: {str(e)}") from e

    def _setup_container(self) -> None:
        """Perform initial container setup."""
        # Install git
        self.execute_command("apt-get update && apt-get install -y git")

        # Install Python dependencies if requirements.txt exists
        if (self.work_dir / "requirements.txt").exists():
            self.execute_command(
                "pip install -r requirements.txt"
            )

    def execute_command(
            self,
            command: str,
            workdir: Optional[str] = None,
            env: Optional[Dict[str, str]] = None
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
            raise EnvironmentError("Container not initialized")

        try:
            exec_result = self.container.exec_run(
                command,
                workdir=workdir or self.config.working_dir,
                environment=env,
            )

            return ExecutionResult(
                exit_code=exec_result.exit_code,
                output=exec_result.output.decode('utf-8')
            )

        except Exception as e:
            raise EnvironmentError(f"Command execution failed: {str(e)}") from e

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
                    shutil.rmtree(self.work_dir)
                except Exception as e:
                    logger.warning(
                        f"Error removing working directory: {str(e)}")

        except Exception as e:
            logger.error(f"Environment cleanup failed: {str(e)}")
            raise EnvironmentError(
                f"Environment cleanup failed: {str(e)}") from e
        finally:
            self.container = None
            self.work_dir = None
            self.repo = None

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
        try:
            # Write changes to files
            for change in changes:
                # TODO: Implement change application logic
                pass

            # Run any necessary build/compile steps
            result = self.execute_command("python setup.py develop")

            if result.exit_code != 0:
                raise EnvironmentError(
                    f"Failed to apply changes: {result.error or result.output}"
                )

            return result

        except Exception as e:
            raise EnvironmentError(f"Failed to apply changes: {str(e)}") from e

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
                f"pytest {' '.join(test_files)} "
                "--cov=. --cov-report=term-missing"
            )

            return self.execute_command(test_command)

        except Exception as e:
            raise EnvironmentError(f"Test execution failed: {str(e)}") from e