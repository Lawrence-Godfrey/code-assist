"""
LLM-powered test generation.

This module implements the TestGenerator class, which uses large language models
to generate comprehensive test cases based on implementation code.
"""

import json
from typing import Dict, List, Optional, Tuple

from code_assistant.logging.logger import get_logger
from code_assistant.models.prompt import PromptModel
from code_assistant.pipeline.coding.models import (
    ChangeType,
    CodeChange,
    FileModification,
)

logger = get_logger(__name__)


class TestGenerator:
    """
    LLM-powered test generation.

    Uses prompt models to generate comprehensive test cases for
    implementations with a focus on coverage and edge cases.
    """

    def __init__(self, prompt_model: PromptModel):
        """
        Initialize the test generator.

        Args:
            prompt_model: The prompt model to use for generation
        """
        self._prompt_model = prompt_model

    def generate_tests(
        self,
        implementation_changes: List[CodeChange],
        test_framework: str = "pytest",
        coverage_target: float = 0.9,
        context: Optional[Dict] = None,
    ) -> List[CodeChange]:
        """
        Generate test cases for implementation changes.

        Args:
            implementation_changes: List of implementation changes
            test_framework: Test framework to use (pytest, unittest)
            coverage_target: Target test coverage (0.0 to 1.0)
            context: Additional context

        Returns:
            List of test file changes
        """
        logger.info(f"Generating tests with {test_framework} framework")

        test_changes = []

        # Process each implementation change to determine what needs tests
        implementation_files = self._extract_implementation_files(
            implementation_changes
        )

        for impl_file in implementation_files:
            # Skip files that shouldn't be tested
            if not impl_file["should_test"]:
                continue

            # Generate test file path
            test_file_path = self._get_test_file_path(
                impl_file["file_path"], test_framework
            )

            # Generate test content for this implementation
            test_content = self._generate_test_content(
                impl_file, test_framework, coverage_target, context
            )

            # Add to test changes
            test_changes.append(
                CodeChange(
                    type=ChangeType.CREATE,
                    file_path=test_file_path,
                    content=test_content,
                )
            )

        logger.info(f"Generated {len(test_changes)} test files")
        return test_changes

    def analyze_coverage(
        self,
        implementation_changes: List[CodeChange],
        test_changes: List[CodeChange],
        context: Optional[Dict] = None,
    ) -> Dict:
        """
        Analyze test coverage for the implementation.

        Args:
            implementation_changes: List of implementation changes
            test_changes: List of test changes
            context: Additional context

        Returns:
            Dictionary with coverage analysis
        """
        logger.info("Analyzing test coverage")

        # Extract implementation and test details
        implementation_files = self._extract_implementation_files(
            implementation_changes
        )
        test_files = {change.file_path: change.content for change in test_changes}

        # System prompt for coverage analysis
        system_prompt = """
        You are an expert in test coverage analysis. Given implementation code and test code,
        analyze the test coverage and identify any gaps.

        Provide your analysis in JSON format with the following structure:
        {
            "overall_coverage_estimate": 0.85,  # estimated coverage as a decimal
            "files": [
                {
                    "file_path": "example.py",
                    "coverage_estimate": 0.9,
                    "covered_elements": ["function_a", "class_B.method_1", ...],
                    "uncovered_elements": ["edge_case_function", ...],
                    "coverage_gaps": ["Exception handling is not tested", ...]
                },
                ...
            ],
            "recommendations": [
                "Add tests for exception handling in file_x.py",
                ...
            ]
        }
        """

        # Create user prompt with implementation and test details
        user_prompt = "## Implementation Files:\n\n"

        for impl_file in implementation_files:
            user_prompt += f"### {impl_file['file_path']}:\n"
            user_prompt += "```python\n"
            user_prompt += impl_file["content"]
            user_prompt += "\n```\n\n"

        user_prompt += "## Test Files:\n\n"

        for test_path, test_content in test_files.items():
            user_prompt += f"### {test_path}:\n"
            user_prompt += "```python\n"
            user_prompt += test_content
            user_prompt += "\n```\n\n"

        user_prompt += "Please analyze the test coverage."

        try:
            # Generate coverage analysis
            response = self._prompt_model.generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,  # Low temperature for consistent analysis
            )

            # Parse JSON response
            analysis = json.loads(response)

            logger.info(
                f"Coverage analysis complete: {analysis['overall_coverage_estimate']:.0%} estimated coverage"
            )
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze coverage: {str(e)}")
            # Return basic analysis as fallback
            return {
                "overall_coverage_estimate": 0.0,
                "files": [],
                "recommendations": ["Could not analyze coverage"],
            }

    def generate_additional_tests(
        self,
        coverage_analysis: Dict,
        implementation_changes: List[CodeChange],
        existing_tests: List[CodeChange],
    ) -> List[CodeChange]:
        """
        Generate additional tests to improve coverage.

        Args:
            coverage_analysis: Coverage analysis from analyze_coverage
            implementation_changes: List of implementation changes
            existing_tests: Existing test changes

        Returns:
            List of additional test changes
        """
        logger.info("Generating additional tests for coverage gaps")

        additional_tests = []
        existing_test_map = {change.file_path: change for change in existing_tests}

        # For each file with coverage gaps
        for file_info in coverage_analysis["files"]:
            # Skip files with good coverage
            if file_info["coverage_estimate"] >= 0.9:
                continue

            # Get implementation file details
            impl_file = next(
                (
                    f
                    for f in self._extract_implementation_files(implementation_changes)
                    if f["file_path"] == file_info["file_path"]
                ),
                None,
            )

            if not impl_file:
                continue

            # Find corresponding test file
            test_file_path = self._get_test_file_path(file_info["file_path"], "pytest")

            # Check if we have this test file already
            if test_file_path in existing_test_map:
                existing_test = existing_test_map[test_file_path]

                # Generate additional tests focusing on gaps
                additional_content = self._generate_gap_tests(
                    impl_file,
                    file_info["uncovered_elements"],
                    file_info["coverage_gaps"],
                    existing_test.content,
                )

                # Create a modified version of the existing test
                additional_tests.append(
                    CodeChange(
                        type=ChangeType.MODIFY,
                        file_path=test_file_path,
                        modifications=[
                            FileModification(
                                type=FileModification.REPLACE,
                                content=additional_content,
                                start_line=1,
                                end_line=999999,
                                # Effectively replace the whole file
                            )
                        ],
                    )
                )
            else:
                # Generate a new test file
                test_content = self._generate_test_content(
                    impl_file,
                    "pytest",
                    0.95,  # Higher coverage target
                    {"focus_on": file_info["uncovered_elements"]},
                )

                additional_tests.append(
                    CodeChange(
                        type=ChangeType.CREATE,
                        file_path=test_file_path,
                        content=test_content,
                    )
                )

        logger.info(f"Generated {len(additional_tests)} additional test files/updates")
        return additional_tests

    def _extract_implementation_files(self, changes: List[CodeChange]) -> List[Dict]:
        """Extract implementation files from code changes."""
        implementation_files = []

        for change in changes:
            # Skip test files and non-Python files
            if change.file_path.startswith("test_") or not change.file_path.endswith(
                ".py"
            ):
                continue

            file_name = change.file_path.split("/")[-1]

            # Determine if this should be tested
            should_test = not file_name.startswith("__") and not file_name.startswith(
                "test_"
            )

            if change.type == ChangeType.CREATE:
                implementation_files.append(
                    {
                        "file_name": file_name,
                        "file_path": change.file_path,
                        "content": change.content,
                        "should_test": should_test,
                    }
                )
            elif change.type == ChangeType.MODIFY and change.modifications:
                # For simplicity, we'll just use the first modification
                implementation_files.append(
                    {
                        "file_name": file_name,
                        "file_path": change.file_path,
                        "content": change.modifications[0].content,
                        "should_test": should_test,
                    }
                )

        return implementation_files

    def _get_test_file_path(self, implementation_path: str, framework: str) -> str:
        """Get the corresponding test file path for an implementation file."""
        # Extract directory and filename
        import os

        directory, filename = os.path.split(implementation_path)

        # Create test filename
        if not filename.startswith("test_"):
            test_filename = f"test_{filename}"
        else:
            test_filename = filename

        # Handle potential pytest directory structure
        if framework == "pytest" and directory:
            test_dir = os.path.join(directory, "tests")
            return os.path.join(test_dir, test_filename)

        # Default case
        return os.path.join(directory, test_filename) if directory else test_filename

    def _generate_test_content(
        self,
        implementation_file: Dict,
        test_framework: str,
        coverage_target: float,
        context: Optional[Dict] = None,
    ) -> str:
        """
        Generate test content for an implementation file.

        Args:
            implementation_file: Implementation file details
            test_framework: Test framework to use
            coverage_target: Target test coverage
            context: Additional context

        Returns:
            Test file content
        """
        # Determine the appropriate framework
        framework_info = {
            "pytest": {
                "import": "import pytest",
                "style": "pytest style with functions",
                "assert": "pytest assertions (assert x == y)",
            },
            "unittest": {
                "import": "import unittest",
                "style": "unittest.TestCase classes",
                "assert": "unittest assertions (self.assertEqual(x, y))",
            },
        }.get(test_framework.lower())

        # Create system prompt for test generation
        system_prompt = f"""
        You are an expert in test-driven development and {test_framework}.
        Write comprehensive, high-quality tests for the given implementation file.

        Your tests should:
        1. Follow {framework_info['style']}
        2. Use {framework_info['assert']}
        3. Cover both normal operation and edge cases
        4. Be well-organized and documented
        5. Target {coverage_target:.0%} code coverage
        6. Include setup and teardown where appropriate

        Only generate the test code. Do not include explanations or markdown formatting.
        """

        # Extract any focus elements from context
        focus_elements = []
        if context and "focus_on" in context:
            focus_elements = context["focus_on"]

        # Create user prompt
        user_prompt = f"""
        Generate complete test file for: {implementation_file['file_path']}

        Implementation code:
        ```python
        {implementation_file['content']}
        ```
        """

        if focus_elements:
            user_prompt += f"\nFocus particularly on testing these elements: {', '.join(focus_elements)}\n"

        user_prompt += (
            f"\nCreate a complete {test_framework} test file with thorough coverage."
        )

        # Generate test content
        response = self._prompt_model.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,  # Slightly higher for test creativity
        )

        # Clean up code blocks if present
        test_content = self._clean_code_blocks(response)

        # Ensure the test contains proper imports
        if test_framework == "pytest" and "import pytest" not in test_content:
            test_content = f"import pytest\n{test_content}"

        if test_framework == "unittest" and "import unittest" not in test_content:
            test_content = f"import unittest\n{test_content}"

        # Import the module being tested
        module_name = (
            implementation_file["file_path"]
            .replace(".py", "")
            .replace("/", ".")
            .strip(".")
        )
        if (
            module_name
            and f"import {module_name}" not in test_content
            and f"from {module_name}" not in test_content
        ):
            if "import " in test_content:
                # Add after the last import
                import_lines = []
                code_lines = []
                for line in test_content.split("\n"):
                    if line.startswith("import ") or line.startswith("from "):
                        import_lines.append(line)
                    else:
                        code_lines.append(line)

                import_lines.append(f"import {module_name}")
                test_content = "\n".join(import_lines + code_lines)
            else:
                test_content = f"import {module_name}\n\n{test_content}"

        return test_content

    def _generate_gap_tests(
        self,
        implementation_file: Dict,
        uncovered_elements: List[str],
        coverage_gaps: List[str],
        existing_test_content: str,
    ) -> str:
        """
        Generate additional tests focusing on coverage gaps.

        Args:
            implementation_file: Implementation file details
            uncovered_elements: List of uncovered elements
            coverage_gaps: List of coverage gaps
            existing_test_content: Existing test content

        Returns:
            Updated test content
        """
        # Create system prompt for gap test generation
        system_prompt = """
        You are an expert in improving test coverage. Given implementation code, coverage gaps,
        and existing tests, extend the test suite to cover the missing elements.

        Your additional tests should:
        1. Maintain the style and structure of the existing tests
        2. Focus specifically on the uncovered elements and gaps
        3. Be thorough and well-documented
        4. Not duplicate existing test cases

        Only provide the complete updated test file. Do not include explanations or markdown formatting.
        """

        # Create user prompt
        user_prompt = f"""
        Implementation code:
        ```python
        {implementation_file['content']}
        ```

        Existing test file:
        ```python
        {existing_test_content}
        ```

        Uncovered elements that need tests:
        {', '.join(uncovered_elements)}

        Coverage gaps to address:
        {', '.join(coverage_gaps)}

        Please extend the existing test file to cover these gaps while maintaining the same style and structure.
        Return the complete updated test file.
        """

        # Generate updated test content
        response = self._prompt_model.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
        )

        # Clean up code blocks if present
        return self._clean_code_blocks(response)

    def _clean_code_blocks(self, content: str) -> str:
        """Remove markdown code blocks if present."""
        # Check if content is wrapped in code blocks
        if content.startswith("```") and content.endswith("```"):
            # Find the first newline after the opening ```
            start_idx = content.find("\n")
            if start_idx != -1:
                # Find the last ``` and extract content between them
                end_idx = content.rfind("```")
                return content[start_idx + 1 : end_idx].strip()

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
