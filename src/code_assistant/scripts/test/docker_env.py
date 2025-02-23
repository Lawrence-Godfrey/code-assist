"""
Test script for Docker execution environment.

This script tests the Docker environment setup, command execution,
and cleanup functionality.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path if needed
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent))

from code_assistant.logging.logger import get_logger
from code_assistant.pipeline.coding.environment.config import DockerConfig
from code_assistant.pipeline.coding.environment.docker import DockerEnvironment

logger = get_logger(__name__)


async def test_docker_environment(repo_url: str):
    """Test Docker environment with basic operations."""
    # Create Docker environment
    logger.info("Creating Docker environment")
    config = DockerConfig(
        base_image="python:3.12",
        memory_limit="1g",
        cpu_limit=1,
    )
    env = DockerEnvironment(config)

    try:
        # Setup environment
        logger.info(f"Setting up environment with repo: {repo_url}")
        await env.setup(repo_url)

        # Execute basic commands
        logger.info("Testing command execution")
        result = env.execute_command("python --version")
        logger.info(f"Python version: {result.output.strip()}")

        result = env.execute_command("ls -la")
        logger.info(f"Directory listing: \n{result.output}")

        # Create test file
        logger.info("Creating initial test file")
        test_content = """
        def test_function():
            message = "Hello, Pipeline!"
            assert message == "Hello, Pipeline!"

        def test_success():
            message = "Hello, Pipeline!"
            assert message == "Hello, Pipeline!"
        """

        result = env.execute_command(f"echo '{test_content}' > test_file.py")
        logger.info(f"Create file result: {result.exit_code}")

        # Run pytest
        logger.info("Installing pytest")
        env.execute_command("pip install pytest")

        logger.info("Running pytest on initial test file")
        result = env.execute_command("pytest -xvs test_file.py")
        logger.info(f"Initial test result: \n{result.output}")

        # Apply changes
        logger.info("Testing apply_changes")
        changes = ["test_file2.py:print('This is another test file')"]
        result = await env.apply_changes(changes)
        logger.info(f"Apply changes result: {result.exit_code}")

        # Verify file was created
        result = env.execute_command("cat test_file2.py")
        logger.info(f"New file content: {result.output}")

        # Run tests
        logger.info("Testing run_tests")
        result = await env.run_tests(["test_file.py"])
        logger.info(f"Run tests result: \n{result.output}")

        logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

    finally:
        # Cleanup
        logger.info("Cleaning up environment")
        await env.cleanup()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Docker execution environment")
    parser.add_argument(
        "--repo",
        default="https://github.com/Lawrence-Godfrey/code-assist.git",
        help="Git repository URL to clone",
    )

    args = parser.parse_args()

    asyncio.run(test_docker_environment(args.repo))
