"""
LLM-powered code generation assistant.

This module implements the CodeAssistant class, which uses large language models
to generate code changes, analyze errors, and create implementation plans based
on requirements.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from code_assistant.logging.logger import get_logger
from code_assistant.models.prompt import PromptModel
from code_assistant.pipeline.coding.models import (
    ChangeType,
    CodeChange,
    FileModification,
    ModificationType
)

logger = get_logger(__name__)


@dataclass
class ImplementationPlan:
    """Structured plan for code implementation."""
    steps: List[str]
    file_changes: List[Dict]
    dependencies: List[str]
    test_approach: List[str]


class Assistant:
    """
    LLM-powered code generation assistant.

    Uses prompt models to generate implementation plans, code changes,
    tests, and error analysis based on requirements.
    """

    def __init__(self, prompt_model: PromptModel):
        """
        Initialize the code assistant.

        Args:
            prompt_model: The prompt model to use for generation
        """
        self._prompt_model = prompt_model

    async def create_implementation_plan(
            self,
            requirements: Dict,
            context: Dict
    ) -> ImplementationPlan:
        """
        Create an implementation plan based on requirements.

        Args:
            requirements: The requirements schema
            context: Additional context (codebase knowledge, etc.)

        Returns:
            An implementation plan with steps and file changes
        """
        logger.info("Creating implementation plan from requirements")

        # Format requirements for the prompt
        req_formatted = self._format_requirements(requirements)

        # Prepare the prompt for plan generation
        system_prompt = """
        You are an expert software developer tasked with creating implementation plans.
        Given the requirements, create a detailed plan for implementation that includes:
        1. Step-by-step implementation steps
        2. File changes needed (new files, modifications)
        3. Dependencies or prerequisites
        4. Testing approach

        Your response must be in JSON format with the following structure:
        {
            "steps": ["Step 1: ...", "Step 2: ...", ...],
            "file_changes": [
                {
                    "file_path": "path/to/file.py",
                    "change_type": "create|modify|delete",
                    "description": "Brief description of changes"
                },
                ...
            ],
            "dependencies": ["Dependency 1", "Dependency 2", ...],
            "test_approach": ["Test step 1", "Test step 2", ...]
        }
        """

        user_prompt = f"""
        Here are the requirements:

        {req_formatted}

        Please create a detailed implementation plan.
        """

        try:
            # Generate the implementation plan
            response = self._prompt_model.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,  # Lower temperature for more consistent plans
            )

            # Parse the JSON response
            plan_data = json.loads(response)

            # Create the implementation plan
            plan = ImplementationPlan(
                steps=plan_data["steps"],
                file_changes=plan_data["file_changes"],
                dependencies=plan_data["dependencies"],
                test_approach=plan_data["test_approach"]
            )

            logger.info(
                f"Created implementation plan with {len(plan.steps)} steps and {len(plan.file_changes)} file changes")
            return plan

        except Exception as e:
            logger.error(f"Failed to create implementation plan: {str(e)}")
            # Return a basic plan as fallback
            return ImplementationPlan(
                steps=["Implement basic functionality"],
                file_changes=[
                    {"file_path": "implementation.py", "change_type": "create",
                     "description": "Basic implementation"}],
                dependencies=["Python standard library"],
                test_approach=["Create unit tests"]
            )

    async def generate_code_changes(
            self,
            plan: ImplementationPlan,
            context: Dict
    ) -> List[CodeChange]:
        """
        Generate concrete code changes based on the implementation plan.

        Args:
            plan: The implementation plan
            context: Additional context (codebase knowledge, etc.)

        Returns:
            List of code changes to apply
        """
        logger.info("Generating code changes from implementation plan")

        code_changes = []

        # Process each file change in the plan
        for file_change in plan.file_changes:
            file_path = file_change["file_path"]
            change_type = file_change["change_type"]

            # Generate code for this file
            file_content = await self._generate_file_content(
                file_path,
                change_type,
                file_change["description"],
                plan,
                context
            )

            # Create appropriate code change object
            if change_type == "create":
                code_changes.append(
                    CodeChange(
                        type=ChangeType.CREATE,
                        file_path=file_path,
                        content=file_content
                    )
                )
            elif change_type == "modify":
                # For modifications, we need to have the original file content
                # This would normally come from the environment or be retrieved
                # For now we'll just create a placeholder modification
                code_changes.append(
                    CodeChange(
                        type=ChangeType.MODIFY,
                        file_path=file_path,
                        modifications=[
                            FileModification(
                                type=ModificationType.REPLACE,
                                content=file_content,
                                start_line=1,
                                end_line=1000  # This is a placeholder
                            )
                        ]
                    )
                )
            elif change_type == "delete":
                code_changes.append(
                    CodeChange(
                        type=ChangeType.DELETE,
                        file_path=file_path
                    )
                )

        logger.info(f"Generated {len(code_changes)} code changes")
        return code_changes

    async def generate_tests(
            self,
            implementation_changes: List[CodeChange],
            plan: ImplementationPlan,
            context: Dict
    ) -> List[CodeChange]:
        """
        Generate test cases for the implementation.

        Args:
            implementation_changes: The implementation changes
            plan: The implementation plan
            context: Additional context

        Returns:
            List of test file changes
        """
        logger.info("Generating tests for implementation")

        test_changes = []

        # Extract implementation details to help with test generation
        implementation_details = self._extract_implementation_details(
            implementation_changes)

        # System prompt for test generation
        system_prompt = """
        You are an expert in test-driven development. Given the implementation details and test approach,
        generate comprehensive test files that validate the implementation.

        Create thorough tests that cover:
        1. Basic functionality
        2. Edge cases
        3. Error conditions

        Use pytest for the test framework. Each test should have clear assertions and comments explaining
        what is being tested. Follow best practices for test organization and readability.
        """

        # Generate tests for each implementation file
        for impl_file in implementation_details:
            if not impl_file.get("should_test", True):
                continue

            test_file_path = f"test_{impl_file['file_name']}"
            if not test_file_path.endswith(".py"):
                test_file_path += ".py"

            # Create user prompt for this specific file
            user_prompt = f"""
            Generate tests for the following implementation:

            File name: {impl_file['file_name']}

            File content:
            ```python
            {impl_file['content']}
            ```

            Test approach:
            {' '.join(plan.test_approach)}

            Generate a complete test file named {test_file_path} that thoroughly tests this implementation.
            """

            # Generate test content
            test_content = self._prompt_model.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Slightly higher for some creativity in tests
            )

            # Clean up any markdown code blocks
            test_content = self._clean_code_blocks(test_content)

            # Add to test changes
            test_changes.append(
                CodeChange(
                    type=ChangeType.CREATE,
                    file_path=test_file_path,
                    content=test_content
                )
            )

        logger.info(f"Generated {len(test_changes)} test files")
        return test_changes

    async def analyze_error(
            self,
            error: Union[str, Exception],
            changes: List[CodeChange],
            test_output: Optional[str] = None,
            context: Dict = None
    ) -> Dict:
        """
        Analyze an error and suggest fixes.

        Args:
            error: The error message or exception
            changes: The code changes that led to the error
            test_output: Optional test output for additional context
            context: Additional context

        Returns:
            Dictionary with analysis and suggested fixes
        """
        logger.info("Analyzing error")

        # Convert error to string if it's an exception
        error_msg = str(error)

        # Format code changes for analysis
        formatted_changes = self._format_code_changes(changes)

        # Create system prompt for error analysis
        system_prompt = """
        You are an expert programmer and debugger. Given code changes and an error message, 
        analyze the error and suggest specific fixes. Think step by step.

        First, identify the type of error and its likely cause.
        Then, suggest specific code changes to fix the issue.

        Your response should be in JSON format with the following structure:
        {
            "error_type": "Syntax error|Logic error|Import error|etc.",
            "error_location": "File and line where error occurred (best guess)",
            "root_cause": "Explanation of the root cause",
            "suggested_fixes": [
                {
                    "file_path": "path to file",
                    "line_number": 123,
                    "original_code": "problematic code",
                    "fixed_code": "fixed code",
                    "explanation": "Why this fixes the issue"
                },
                ...
            ],
            "additional_recommendations": ["Recommendation 1", "Recommendation 2", ...]
        }
        """

        # Create user prompt with error details
        user_prompt = f"""
        The following code changes resulted in an error:

        {formatted_changes}

        Error message:
        {error_msg}
        """

        # Add test output if available
        if test_output:
            user_prompt += f"\n\nTest output:\n{test_output}\n"

        try:
            # Generate error analysis
            response = self._prompt_model.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,  # Low temperature for consistent analysis
            )

            # Parse JSON response
            analysis = json.loads(response)

            logger.info(f"Completed error analysis: {analysis['error_type']}")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze error: {str(e)}")
            # Return basic analysis as fallback
            return {
                "error_type": "Unknown error",
                "error_location": "Unknown",
                "root_cause": "Could not determine root cause",
                "suggested_fixes": [],
                "additional_recommendations": ["Review code manually"]
            }

    def _format_requirements(self, requirements: Dict) -> str:
        """Format requirements for prompt input."""
        if not requirements:
            return "No requirements provided."

        result = []

        # Handle task type
        if task_type := requirements.get("task_type"):
            result.append(
                f"Task Type: {task_type.value if hasattr(task_type, 'value') else task_type}")

        # Handle description
        if description := requirements.get("description"):
            result.append(f"Description:\n{description}")

        # Handle definition of done (DoD)
        if dod := requirements.get("dod"):
            dod_items = "\n".join([f"- {item}" for item in dod])
            result.append(f"Definition of Done:\n{dod_items}")

        # Handle risk and effort levels
        if risk := requirements.get("risk"):
            result.append(
                f"Risk Level: {risk.value if hasattr(risk, 'value') else risk}")

        if effort := requirements.get("effort"):
            result.append(
                f"Effort Level: {effort.value if hasattr(effort, 'value') else effort}")

        # Handle focus region
        if focus_region := requirements.get("focus_region"):
            result.append(f"Focus Region: {focus_region}")

        return "\n\n".join(result)

    async def _generate_file_content(
            self,
            file_path: str,
            change_type: str,
            description: str,
            plan: ImplementationPlan,
            context: Dict
    ) -> str:
        """Generate content for a file based on the implementation plan."""
        # Determine file type from path
        file_extension = file_path.split(".")[-1] if "." in file_path else ""
        file_type = self._get_file_type(file_extension)

        # Create system prompt for code generation
        system_prompt = f"""
        You are an expert Python developer. Create high-quality, well-documented {file_type} code
        based on the given description and implementation plan.

        Your code should follow these guidelines:
        1. Be clean, readable, and follow PEP 8 guidelines
        2. Include proper docstrings and comments
        3. Handle errors appropriately
        4. Be efficient and follow best practices
        5. Be complete and self-contained (unless importing other modules)
        """

        # Create user prompt with file details
        user_prompt = f"""
        Please generate the complete contents for the file: {file_path}

        File description: {description}

        Implementation steps:
        {' '.join(plan.steps)}

        The file should be a complete, well-structured {file_type} file ready for use.
        Only respond with the code content, no additional explanations or markdown formatting.
        """

        # Generate file content
        content = self._prompt_model.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
        )

        # Clean up any markdown code blocks
        return self._clean_code_blocks(content)

    def _get_file_type(self, extension: str) -> str:
        """Determine file type from extension."""
        file_types = {
            "py": "Python",
            "js": "JavaScript",
            "ts": "TypeScript",
            "html": "HTML",
            "css": "CSS",
            "md": "Markdown",
            "json": "JSON",
            "yaml": "YAML",
            "yml": "YAML",
            "sh": "Shell",
            "txt": "text",
        }

        return file_types.get(extension.lower(), "code")

    def _clean_code_blocks(self, content: str) -> str:
        """Remove markdown code blocks if present."""
        # Check if content is wrapped in code blocks
        if content.startswith("```") and content.endswith("```"):
            # Find the first newline after the opening ```
            start_idx = content.find("\n")
            if start_idx != -1:
                # Find the last ``` and extract content between them
                end_idx = content.rfind("```")
                return content[start_idx + 1:end_idx].strip()

        # Also handle cases where code blocks are in the middle
        if "```python" in content or "```" in content:
            lines = content.split("\n")
            in_code_block = False
            code_lines = []

            for line in lines:
                if line.startswith("```"):
                    in_code_block = not in_code_block
                    continue

                if in_code_block:
                    code_lines.append(line)

            if code_lines:
                return "\n".join(code_lines)

        # If no code blocks found, return the original content
        return content

    def _extract_implementation_details(self, changes: List[CodeChange]) -> \
    List[Dict]:
        """Extract implementation details from code changes."""
        details = []

        for change in changes:
            if change.type == ChangeType.CREATE:
                # Extract file name from path
                file_name = change.file_path.split("/")[-1]

                # Determine if this file should have tests
                should_test = (
                        file_name.endswith(".py") and
                        not file_name.startswith("__") and
                        not file_name.startswith("test_")
                )

                details.append({
                    "file_name": file_name,
                    "file_path": change.file_path,
                    "content": change.content,
                    "should_test": should_test
                })

            elif change.type == ChangeType.MODIFY and change.modifications:
                # For simplicity, we'll just use the first modification
                file_name = change.file_path.split("/")[-1]

                # Consider if this should be tested
                should_test = (
                        file_name.endswith(".py") and
                        not file_name.startswith("__") and
                        not file_name.startswith("test_")
                )

                details.append({
                    "file_name": file_name,
                    "file_path": change.file_path,
                    "content": change.modifications[0].content,
                    "should_test": should_test
                })

        return details

    def _format_code_changes(self, changes: List[CodeChange]) -> str:
        """Format code changes for error analysis."""
        result = []

        for change in changes:
            result.append(f"File: {change.file_path}")
            result.append(f"Change Type: {change.type.value}")

            if change.type == ChangeType.CREATE:
                result.append("Content:")
                result.append("```python")
                result.append(change.content)
                result.append("```")

            elif change.type == ChangeType.MODIFY and change.modifications:
                for i, mod in enumerate(change.modifications):
                    result.append(f"Modification {i + 1}:")
                    result.append(f"Type: {mod.type.value}")
                    result.append(
                        f"Lines: {mod.start_line or 'N/A'} to {mod.end_line or 'N/A'}")
                    result.append("Content:")
                    result.append("```python")
                    result.append(mod.content)
                    result.append("```")

            result.append("-" * 50)

        return "\n".join(result)