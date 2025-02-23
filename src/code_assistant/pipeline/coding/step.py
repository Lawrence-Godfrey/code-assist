"""
Pipeline step for code coding.

This module implements the coding step in the pipeline, responsible for
setting up the environment, applying code changes, and running tests.
"""

import os
from typing import Dict, List, Optional

from code_assistant.feedback.manager import FeedbackManager
from code_assistant.feedback.mixins import FeedbackEnabled
from code_assistant.logging.logger import get_logger
from code_assistant.models.prompt import PromptModel
from code_assistant.pipeline.coding.environment.base import ExecutionResult
from code_assistant.pipeline.coding.environment.exceptions import EnvironmentError
from code_assistant.pipeline.coding.environment.factory import EnvironmentFactory
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
    ):
        """
        Initialize the execution step.

        Args:
            prompt_model: The prompt model for code generation
            feedback_manager: The feedback manager for user interaction
            env_type: Type of execution environment ("docker", "venv")
            env_config: Optional configuration for environment
        """
        PipelineStep.__init__(self)
        FeedbackEnabled.__init__(self, feedback_manager)

        self._prompt_model = prompt_model
        self._environment = EnvironmentFactory.create_environment(env_type, env_config)

    async def execute(self, context: Dict) -> None:
        """
        Execute the pipeline step.

        Args:
            context: Pipeline context containing the task information

        Raises:
            ValueError: If required context is missing
        """
        repo_url = os.getenv(
            "REPO_URL"
        )  # TODO: Settings object is probably needed soon
        branch = (
            "agent/feature_test"  # TODO: Need to generate branch names automatically
        )

        try:
            # Setup environment
            logger.info("Setting up execution environment")
            await self._environment.setup(repo_url, branch)

            # Basic implementation - apply simple changes
            changes = await self._generate_changes(context)

            # Apply changes and run tests
            result = await self._apply_and_test(changes)

            # Update context with results
            context["execution_results"] = {
                "success": result.success,
                "output": result.output,
                "error": result.error,
            }

            # Request feedback
            if not result.success:
                response = self.request_step_feedback(
                    context="execution_error",
                    prompt=(
                        f"Execution failed with error:\n{result.error or result.output}\n\n"
                        "Would you like to retry with different changes? (yes/no)"
                    ),
                )

                if response.lower().strip() in ("y", "yes"):
                    # TODO: Implement retry logic
                    pass

        except Exception as e:
            logger.error(f"Execution step failed: {str(e)}")
            raise

        finally:
            # Always clean up
            await self._cleanup()

        # Continue to next step
        return self.execute_next(context)

    async def _generate_changes(self, context: Dict) -> List[str]:
        """
        Generate code changes based on context.

        Args:
            context: Pipeline context

        Returns:
            List of code changes
        """
        # Simple placeholder implementation
        # In reality, this would use the prompt model to generate changes
        logger.info("Generating code changes")

        # For testing, we'll create a simple file
        return ["test_file.py:print('Hello, Pipeline!')"]

    async def _apply_and_test(self, changes: List[str]) -> ExecutionResult:
        """
        Apply changes and run tests.

        Args:
            changes: List of code changes

        Returns:
            ExecutionResult with test results
        """
        # Apply changes
        logger.info(f"Applying {len(changes)} changes")
        apply_result = await self._environment.apply_changes(changes)

        if not apply_result.success:
            logger.error(
                f"Failed to apply changes: {apply_result.error or apply_result.output}"
            )
            return apply_result

        # Run tests
        logger.info("Running tests")
        return await self._environment.run_tests(["test_file.py"])

    async def _cleanup(self) -> None:
        """Clean up environment resources."""
        try:
            logger.info("Cleaning up execution environment")
            await self._environment.cleanup()
        except Exception as e:
            logger.error(f"Environment cleanup failed: {str(e)}")
