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
from code_assistant.pipeline.coding.models import (
    ChangeType,
    CodeChange,
    FileModification,
    ModificationType,
)

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
        test_content = (
            "def test_function():\n"
            '    message = "Hello, Pipeline!"\n'
            '    assert message == "Hello, Pipeline!"\n'
            "\n"
            "def test_success():\n"
            '    message = "Hello, Pipeline!"\n'
            '    assert message == "Hello, Pipeline!"\n'
        )

        result = env.execute_command(f"echo '{test_content}' > test_file.py")
        logger.info(f"Create file result: {result.exit_code}")

        # Run pytest
        logger.info("Installing pytest")
        env.execute_command("pip install pytest")

        logger.info("Running pytest on initial test file")
        result = env.execute_command("pytest -xvs test_file.py")
        logger.info(f"Initial test result: \n{result.output}")

        # Test all types of changes
        logger.info("Testing all types of changes")

        # 1. CREATE new file
        create_change = CodeChange(
            type=ChangeType.CREATE,
            file_path="new_test.py",
            content="def test_new():\n    assert True\n",
        )

        # 2. MODIFY with INSERT
        insert_modification = FileModification(
            type=ModificationType.INSERT,
            content="def test_inserted():\n    assert True\n",
            start_line=1,
        )

        modify_insert_change = CodeChange(
            type=ChangeType.MODIFY,
            file_path="test_file.py",
            modifications=[insert_modification],
        )

        # 3. MODIFY with REPLACE
        replace_modification = FileModification(
            type=ModificationType.REPLACE,
            content="def test_replaced():\n    assert True\n",
            start_line=1,
            end_line=3,
        )

        modify_replace_change = CodeChange(
            type=ChangeType.MODIFY,
            file_path="test_file.py",
            modifications=[replace_modification],
        )

        # 4. MODIFY with DELETE
        delete_modification = FileModification(
            type=ModificationType.DELETE, content="", start_line=6, end_line=8
        )

        modify_delete_change = CodeChange(
            type=ChangeType.MODIFY,
            file_path="test_file.py",
            modifications=[delete_modification],
        )

        # 5. DELETE file
        delete_change = CodeChange(type=ChangeType.DELETE, file_path="new_test.py")

        # Apply and verify each change
        async def test_change(change: CodeChange, description: str):
            logger.info(f"Testing {description}")
            result = await env.apply_changes([change])
            logger.info(f"Apply {description} result: {result.exit_code}")

            # Verify the change
            if change.type == ChangeType.DELETE:
                result = env.execute_command(
                    f"test -e {change.file_path} && echo 'File exists' || echo 'File does not exist'"
                )
                logger.info(f"Verify deletion: {result.output}")
            else:
                result = env.execute_command(f"cat {change.file_path}")
                logger.info(f"File content after {description}:\n{result.output}")

            # Run tests after each change
            result = env.execute_command("pytest -xvs")
            logger.info(f"Test result after {description}:\n{result.output}")

        # Execute all changes in sequence
        changes_to_test = [
            (create_change, "CREATE new file"),
            (modify_insert_change, "MODIFY with INSERT"),
            (modify_replace_change, "MODIFY with REPLACE"),
            (modify_delete_change, "MODIFY with DELETE"),
            (delete_change, "DELETE file"),
        ]

        for change, description in changes_to_test:
            await test_change(change, description)

        logger.info("All change types tested")

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
