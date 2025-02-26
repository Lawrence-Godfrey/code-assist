"""
Pipeline step for code coding with LLM integration.

This module implements the coding step in the pipeline, responsible for
setting up the environment, generating and applying code changes, and running tests.
The step leverages LLM capabilities for code generation, testing, and error handling.
"""

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from code_assistant.feedback.manager import FeedbackManager
from code_assistant.feedback.mixins import FeedbackEnabled
from code_assistant.logging.logger import get_logger
from code_assistant.models.prompt import PromptModel
from code_assistant.pipeline.coding.checkpoints.manager import CheckpointManager
from code_assistant.pipeline.coding.checkpoints.models import Checkpoint, \
    CheckpointStatus
from code_assistant.pipeline.coding.environment.base import \
    ExecutionEnvironment, ExecutionResult
from code_assistant.pipeline.coding.environment.exceptions import \
    EnvironmentError
from code_assistant.pipeline.coding.environment.factory import \
    EnvironmentFactory
from code_assistant.pipeline.coding.llm.assistant import Assistant, \
    ImplementationPlan
from code_assistant.pipeline.coding.llm.error_analyzer import ErrorAnalyzer
from code_assistant.pipeline.coding.llm.test_generator import TestGenerator
from code_assistant.pipeline.coding.models import ChangeResult, ChangeType, \
    CodeChange, FileModification
from code_assistant.pipeline.step import PipelineStep

logger = get_logger(__name__)


class CodingStep(PipelineStep, FeedbackEnabled):
    """Pipeline step for code execution with LLM integration."""

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
        self._environment = EnvironmentFactory.create_environment(env_type,
                                                                  env_config)
        self._checkpoint_manager = CheckpointManager()
        self._max_retry_attempts = max_retry_attempts
        self._current_attempt = 0
        self._branch_name = None

        # Initialize LLM components
        self._code_assistant = Assistant(prompt_model)
        self._test_generator = TestGenerator(prompt_model)
        self._error_analyzer = ErrorAnalyzer(prompt_model)

        # Store the implementation plan and changes for reference
        self._implementation_plan = None
        self._implementation_changes = []
        self._test_changes = []

    async def execute(self, context: Dict) -> None:
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
        self._branch_name = self._generate_branch_name(
            context["requirements_schema"])

        try:
            # Setup environment
            logger.info("Setting up execution environment")
            await self._environment.setup(repo_url, self._branch_name)

            # Initialize git config
            await self._initialize_git_config()

            # Start implementation process
            await self._implement_and_test(context)

            # Push changes
            await self._push_changes()

            # Request feedback on implementation
            await self._request_implementation_feedback(context)

        except Exception as e:
            # Handle the error
            logger.error(f"Execution step failed: {str(e)}")
            await self._handle_execution_error(e, context)
            raise

        finally:
            # Clean up environment
            await self._cleanup()

        # Continue to next step if successful
        return self.execute_next(context)

    async def _implement_and_test(self, context: Dict) -> None:
        """
        Implement and test code changes based on requirements.

        Args:
            context: Pipeline context
        """
        # Generate implementation plan
        self._implementation_plan = await self._code_assistant.create_implementation_plan(
            context["requirements_schema"], context
        )

        # Present plan to user for approval
        plan_approved = await self._request_plan_approval(
            self._implementation_plan)

        if not plan_approved:
            # If plan is not approved, regenerate with feedback
            feedback = self.request_step_feedback(
                context="implementation_plan_feedback",
                prompt="What changes would you like to make to the implementation plan?"
            )

            # Update context with feedback
            context["plan_feedback"] = feedback

            # Regenerate plan
            self._implementation_plan = await self._code_assistant.create_implementation_plan(
                context["requirements_schema"], context
            )

            # Check approval again
            plan_approved = await self._request_plan_approval(
                self._implementation_plan)

            if not plan_approved:
                raise ValueError(
                    "Implementation plan was not approved after revision")

        # Generate code changes
        self._implementation_changes = await self._code_assistant.generate_code_changes(
            self._implementation_plan, context
        )

        # Generate tests
        self._test_changes = await self._test_generator.generate_tests(
            self._implementation_changes,
            test_framework="pytest",
            coverage_target=0.9,
            context=context
        )

        # Combine implementation and test changes
        all_changes = self._implementation_changes + self._test_changes

        # Apply changes
        logger.info(f"Applying {len(all_changes)} code changes")
        apply_result = await self._apply_changes(all_changes)

        if not apply_result.success:
            # If failed to apply changes, analyze and fix
            await self._handle_apply_error(apply_result, all_changes, context)

        # Create checkpoint for implementation
        impl_checkpoint = await self._checkpoint_manager.create_checkpoint(
            self._environment,
            self._generate_commit_message(context, all_changes),
            all_changes
        )

        # Run tests
        test_result = await self._run_tests(impl_checkpoint)

        # If tests failed, analyze and fix
        if not test_result.success:
            await self._handle_test_failure(test_result, all_changes, context)

        # Update context with results
        context["coding_results"] = {
            "success": True,
            "changes": [str(c) for c in all_changes],
            "plan": self._implementation_plan.__dict__,
            "branch_name": self._branch_name,
            "checkpoint_id": impl_checkpoint.id,
            "commit_hash": impl_checkpoint.commit_hash,
        }

    async def _handle_apply_error(
            self,
            apply_result: ExecutionResult,
            changes: List[CodeChange],
            context: Dict
    ) -> None:
        """
        Handle errors when applying changes.

        Args:
            apply_result: Result of applying changes
            changes: List of changes that failed
            context: Pipeline context
        """
        logger.info("Analyzing apply error and generating fixes")

        # Analyze error
        error_analysis = await self._error_analyzer.analyze_error(
            apply_result.error or apply_result.output,
            changes
        )

        # Generate fixes
        fix_changes = await self._error_analyzer.generate_fixes(error_analysis,
                                                                changes)

        # Apply fixes
        if fix_changes:
            logger.info(f"Applying {len(fix_changes)} fixes for apply error")
            fix_result = await self._apply_changes(fix_changes)

            if not fix_result.success:
                # If still failing, request manual intervention
                response = self.request_step_feedback(
                    context="apply_error_intervention",
                    prompt=(
                        f"Failed to apply fixes. Error:\n\n"
                        f"{fix_result.error or fix_result.output}\n\n"
                        f"Analysis:\n{error_analysis['root_cause']}\n\n"
                        "Would you like to manually fix the issues? (yes/no)"
                    )
                )

                if response.lower().strip() in ("y", "yes"):
                    # Prompt for manual fixes
                    manual_fix = self.request_step_feedback(
                        context="manual_code_fix",
                        prompt=(
                            "Please provide the corrected code for the files that need fixing. "
                            "Format your response as: [filename]\n[code]\n[END]\n\nYou can include "
                            "multiple files separated by [END]."
                        )
                    )

                    # Apply manual fixes
                    await self._apply_manual_fixes(manual_fix)
                else:
                    raise ValueError(
                        f"Cannot proceed with implementation due to unresolved errors: {apply_result.error}")
        else:
            # No fixes generated, request manual intervention
            raise ValueError(
                f"Failed to apply changes and couldn't generate fixes: {apply_result.error}")

    async def _handle_test_failure(
            self,
            test_result: ExecutionResult,
            changes: List[CodeChange],
            context: Dict
    ) -> None:
        """
        Handle test failures.

        Args:
            test_result: Test execution result
            changes: Code changes
            context: Pipeline context
        """
        logger.info("Analyzing test failure and generating fixes")

        # Analyze error
        error_analysis = await self._error_analyzer.analyze_error(
            test_result.error or test_result.output,
            changes,
            test_output=test_result.output
        )

        # Present analysis to user
        response = self.request_step_feedback(
            context="test_failure_analysis",
            prompt=(
                    f"Tests failed with the following analysis:\n\n"
                    f"Error type: {error_analysis['error_type']}\n"
                    f"Root cause: {error_analysis['root_cause']}\n\n"
                    "Recommended fixes:\n" +
                    "\n".join([f"- {fix['explanation']}" for fix in
                               error_analysis.get('suggested_fixes', [])]) +
                    "\n\nWould you like to apply these fixes? (yes/no)"
            )
        )

        if response.lower().strip() in ("y", "yes"):
            # Generate and apply fixes
            fix_changes = await self._error_analyzer.generate_fixes(
                error_analysis, changes)

            if fix_changes:
                logger.info(
                    f"Applying {len(fix_changes)} fixes for test failures")
                fix_result = await self._apply_changes(fix_changes)

                if fix_result.success:
                    # Create checkpoint for fixes
                    fix_checkpoint = await self._checkpoint_manager.create_checkpoint(
                        self._environment,
                        f"Fix test failures: {error_analysis['error_type']}",
                        fix_changes
                    )

                    # Run tests again
                    retest_result = await self._run_tests(fix_checkpoint)

                    if not retest_result.success:
                        # If still failing, recurse with a counter to prevent infinite loops
                        context["test_fix_attempts"] = context.get(
                            "test_fix_attempts", 0) + 1

                        if context[
                            "test_fix_attempts"] < 3:  # Limit recursive attempts
                            await self._handle_test_failure(retest_result,
                                                            changes + fix_changes,
                                                            context)
                        else:
                            raise ValueError(
                                "Test failures persist after multiple fix attempts")
                else:
                    # Failed to apply fixes
                    raise ValueError(
                        f"Failed to apply test fixes: {fix_result.error}")
            else:
                # No fixes generated
                raise ValueError("Could not generate fixes for test failures")
        else:
            # User chose not to apply fixes
            raise ValueError("Test failures not fixed per user request")

    async def _apply_manual_fixes(self, manual_fix_text: str) -> None:
        """
        Apply manual fixes provided by the user.

        Args:
            manual_fix_text: Text containing manual fixes in the format:
                             [filename]
                             [code]
                             [END]
        """
        # Parse the manual fix text
        sections = manual_fix_text.split("[END]")

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Find the filename
            filename_match = section.split("\n", 1)
            if len(filename_match) < 2:
                logger.warning(
                    f"Skipping invalid manual fix section: {section}")
                continue

            filename = filename_match[0].strip("[] \t")
            code = filename_match[1].strip()

            if not filename or not code:
                continue

            # Create a change for this file
            manual_change = CodeChange(
                type=ChangeType.CREATE,
                file_path=filename,
                content=code
            )

            # Apply the change
            await self._apply_changes([manual_change])

    async def _request_plan_approval(self, plan: ImplementationPlan) -> bool:
        """
        Request user approval for the implementation plan.

        Args:
            plan: The implementation plan

        Returns:
            Whether the plan was approved
        """
        # Format the plan for display
        plan_display = (
            "## Implementation Plan\n\n"
            "### Steps\n"
        )

        for i, step in enumerate(plan.steps, 1):
            plan_display += f"{i}. {step}\n"

        plan_display += "\n### File Changes\n"

        for file_change in plan.file_changes:
            plan_display += f"- {file_change['file_path']} ({file_change['change_type']}): {file_change['description']}\n"

        plan_display += "\n### Dependencies\n"

        for dependency in plan.dependencies:
            plan_display += f"- {dependency}\n"

        plan_display += "\n### Testing Approach\n"

        for test_step in plan.test_approach:
            plan_display += f"- {test_step}\n"

        # Request feedback
        response = self.request_step_feedback(
            context="implementation_plan_approval",
            prompt=(
                f"{plan_display}\n\n"
                "Do you approve this implementation plan? (yes/no)"
            )
        )

        return response.lower().strip() in ("y", "yes")

    async def _generate_changes(self, context: Dict) -> List[CodeChange]:
        """
        Generate code changes based on requirements.

        Args:
            context: Pipeline context with requirements schema

        Returns:
            List of code changes to apply
        """
        # This method is now handled by the CodeAssistant
        # We'll keep it for backward compatibility but delegate
        return await self._code_assistant.generate_code_changes(
            self._implementation_plan or await self._code_assistant.create_implementation_plan(
                context["requirements_schema"], context
            ),
            context
        )

    async def _apply_changes(self,
                             changes: List[CodeChange]) -> ExecutionResult:
        """
        Apply code changes to the environment.

        Args:
            changes: List of code changes to apply

        Returns:
            Result of applying the changes
        """
        logger.info(f"Applying {len(changes)} code changes")

        # Stage files for commit
        result = await self._environment.apply_changes(changes)

        if result.success:
            # Stage all changes
            stage_result = await self._environment.execute_command("git add .")
            if stage_result.exit_code != 0:
                return ExecutionResult(
                    exit_code=stage_result.exit_code,
                    output=stage_result.output,
                    error=f"Failed to stage changes: {stage_result.error or stage_result.output}"
                )

        return result

    async def _run_tests(self, checkpoint: Checkpoint) -> ExecutionResult:
        """
        Run tests for the current implementation.

        Args:
            checkpoint: The checkpoint to validate with tests

        Returns:
            Result of test execution
        """
        logger.info("Running tests")

        # Find all test files
        find_tests_result = await self._environment.execute_command(
            "find . -name 'test_*.py' -type f -not -path '*/\\.*'"
        )

        if find_tests_result.exit_code != 0:
            # Fall back to our known test files
            test_files = [c.file_path for c in self._test_changes]
        else:
            test_files = find_tests_result.output.strip().split("\n")
            # Remove empty entries and relative path prefix
            test_files = [f.strip("./") for f in test_files if f.strip()]

        if not test_files:
            logger.warning("No test files found")
            return ExecutionResult(
                exit_code=0,
                output="No test files found, skipping tests"
            )

        # Run tests and validate checkpoint
        try:
            await self._checkpoint_manager.validate_checkpoint(
                checkpoint.id,
                self._environment,
                test_files
            )
        except Exception as e:
            logger.error(f"Error validating checkpoint: {str(e)}")
            return ExecutionResult(
                exit_code=1,
                output="",
                error=str(e)
            )

        # Get updated checkpoint with test results
        updated_checkpoint = self._checkpoint_manager.get_checkpoint(
            checkpoint.id)

        if not updated_checkpoint or not updated_checkpoint.test_results:
            return ExecutionResult(
                exit_code=1,
                output="",
                error="Failed to retrieve test results"
            )

        # Convert to ExecutionResult for consistency
        return ExecutionResult(
            exit_code=updated_checkpoint.test_results.exit_code,
            output=updated_checkpoint.test_results.output,
            error=updated_checkpoint.test_results.error
        )

    async def _push_changes(self) -> ExecutionResult:
        """
        Push changes to the remote repository.

        Returns:
            Result of the push operation
        """
        logger.info(f"Pushing changes to branch: {self._branch_name}")

        # Push to remote
        return await self._environment.execute_command(
            f"git push -u origin {self._branch_name}")

    async def _handle_execution_error(self, error: Exception,
                                      context: Dict) -> None:
        """
        Handle errors that occur during execution.

        Args:
            error: The exception that occurred
            context: Pipeline context
        """
        self._current_attempt += 1

        # Determine if we should retry
        if self._current_attempt <= self._max_retry_attempts:
            logger.info(
                f"Retrying execution (attempt {self._current_attempt} of {self._max_retry_attempts})")

            # Get the latest successful checkpoint if any
            latest_valid_checkpoint = None
            for checkpoint in self._checkpoint_manager.get_checkpoints_by_status(
                    CheckpointStatus.VALIDATED):
                if not latest_valid_checkpoint or checkpoint.timestamp > latest_valid_checkpoint.timestamp:
                    latest_valid_checkpoint = checkpoint

            # Roll back if we have a valid checkpoint
            if latest_valid_checkpoint:
                await self._checkpoint_manager.rollback_to_checkpoint(
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
                await self.execute(context)
            else:
                # User doesn't want to retry, cleanup and continue
                await self._cleanup()
        else:
            # Max attempts reached, cleanup and continue
            logger.warning(
                f"Max retry attempts ({self._max_retry_attempts}) reached, giving up")

            # Update context with failure info
            context["coding_results"] = {
                "success": False,
                "error": str(error),
                "attempts": self._current_attempt,
            }

    async def _initialize_git_config(self) -> None:
        """Initialize git configuration in the environment."""
        # Set up Git config for commits
        await self._environment.execute_command(
            "git config --local user.email 'code-assistant@example.com'"
        )
        await self._environment.execute_command(
            "git config --local user.name 'Code Assistant'"
        )

    async def _cleanup(self) -> None:
        """Clean up resources after execution."""
        try:
            logger.info("Cleaning up execution environment")
            await self._environment.cleanup()
        except Exception as e:
            logger.error(f"Environment cleanup failed: {str(e)}")

    async def _request_implementation_feedback(self, context: Dict) -> None:
        """
        Request feedback on the implementation.

        Args:
            context: Pipeline context
        """
        # Get the latest checkpoint
        checkpoint = self._checkpoint_manager.get_latest_checkpoint()
        if not checkpoint:
            logger.warning("No checkpoints found for implementation feedback")
            return

        # Get the checkpoint with test results
        if not checkpoint.test_results:
            logger.warning("No test results found for implementation feedback")
            return

        # Generate implementation summary
        summary = (
            f"## Implementation Complete\n\n"
            f"Branch: `{self._branch_name}`\n"
            f"Commit: `{checkpoint.commit_hash[:8]}`\n\n"
            f"### Test Results\n"
            f"Status: {'✅ PASSED' if checkpoint.test_results.success else '❌ FAILED'}\n"
            f"Tests passed: {checkpoint.test_results.passed_count}\n"
            f"Tests failed: {checkpoint.test_results.failed_count}\n"
            f"Test errors: {checkpoint.test_results.error_count}\n\n"
            f"### Implementation Plan\n"
        )

        if self._implementation_plan:
            for i, step in enumerate(self._implementation_plan.steps, 1):
                summary += f"{i}. {step}\n"

        summary += "\n### Changes Implemented\n"

        # Group changes by type
        change_types = {}
        for change in self._implementation_changes + self._test_changes:
            if change.type.value not in change_types:
                change_types[change.type.value] = []
            change_types[change.type.value].append(change.file_path)

        # Format changes
        for change_type, files in change_types.items():
            summary += f"\n{change_type.capitalize()}d files:\n"
            for file in files:
                summary += f"- {file}\n"

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
            logger.info(
                "User is not satisfied with the implementation. Consider additional changes.")

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

    def _generate_commit_message(self, context: Dict,
                                 changes: List[CodeChange]) -> str:
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
                details.append(
                    f"created {len(file_types[ChangeType.CREATE.value])} files")
            if ChangeType.MODIFY.value in file_types:
                details.append(
                    f"modified {len(file_types[ChangeType.MODIFY.value])} files")
            if ChangeType.DELETE.value in file_types:
                details.append(
                    f"deleted {len(file_types[ChangeType.DELETE.value])} files")

            message += ", ".join(details)

        return message