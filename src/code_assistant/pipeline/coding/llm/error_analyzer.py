"""
LLM-powered error analysis and debugging.

This module implements the ErrorAnalyzer class, which uses large language models
to diagnose errors, suggest fixes, and track error patterns over time.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
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
class ErrorRecord:
    """Record of an error occurrence and resolution attempt."""
    error_message: str
    error_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Optional[Dict] = None
    resolution_attempt: Optional[Dict] = None
    resolved: bool = False


@dataclass
class SuggestedFix:
    """Suggested fix for an error."""
    file_path: str
    line_number: Optional[int]
    original_code: str
    fixed_code: str
    explanation: str


class ErrorAnalyzer:
    """
    LLM-powered error analysis and debugging.

    Diagnoses errors, suggests fixes, and tracks error patterns to improve
    debugging effectiveness over time.
    """

    def __init__(self, prompt_model: PromptModel):
        """
        Initialize the error analyzer.

        Args:
            prompt_model: The prompt model to use for analysis
        """
        self._prompt_model = prompt_model
        self._error_history: List[ErrorRecord] = []

    async def analyze_error(
            self,
            error: Union[str, Exception],
            changes: List[CodeChange],
            test_output: Optional[str] = None,
            code_context: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze an error and suggest fixes.

        Args:
            error: Error message or exception
            changes: Code changes related to the error
            test_output: Test output providing more context
            code_context: Additional context about the codebase

        Returns:
            Analysis results with suggested fixes
        """
        logger.info("Analyzing error")

        # Convert error to string if it's an exception
        error_msg = str(error)

        # Check if we've seen this error before
        similar_errors = self._find_similar_errors(error_msg)
        previous_resolutions = [e.resolution_attempt for e in similar_errors if
                                e.resolved]

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
                    "line_number": 123,  # optional, can be null
                    "original_code": "problematic code",
                    "fixed_code": "fixed code",
                    "explanation": "Why this fixes the issue"
                },
                ...
            ],
            "additional_recommendations": ["Recommendation 1", "Recommendation 2", ...]
        }
        """

        # Add information about previous resolutions if available
        if previous_resolutions:
            system_prompt += "\n\nThis error has been seen before. Previous resolution attempts include:"
            for i, resolution in enumerate(previous_resolutions):
                system_prompt += f"\nAttempt {i + 1}: {resolution}"

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

            # Record this error for future reference
            self._record_error(error_msg, analysis)

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

    async def generate_fixes(
            self,
            analysis: Dict,
            changes: List[CodeChange]
    ) -> List[CodeChange]:
        """
        Generate code changes to fix the identified error.

        Args:
            analysis: Error analysis results
            changes: Original code changes

        Returns:
            List of code changes to fix the error
        """
        logger.info("Generating fixes for error")

        # Map original changes by file path
        change_map = {change.file_path: change for change in changes}

        # Create new changes based on suggested fixes
        fix_changes = []

        for suggested_fix in analysis.get("suggested_fixes", []):
            file_path = suggested_fix.get("file_path")

            if not file_path:
                continue

            original_code = suggested_fix.get("original_code", "")
            fixed_code = suggested_fix.get("fixed_code", "")

            if file_path in change_map:
                # Modify existing file
                original_change = change_map[file_path]

                if original_change.type == ChangeType.CREATE:
                    # For CREATE changes, we can just replace the entire content
                    # if it's a small fix, or create a specific modification
                    if self._is_small_diff(original_change.content, fixed_code):
                        # Complete replacement for small changes
                        fix_changes.append(
                            CodeChange(
                                type=ChangeType.CREATE,
                                file_path=file_path,
                                content=self._replace_substring(
                                    original_change.content,
                                    original_code,
                                    fixed_code
                                )
                            )
                        )
                    else:
                        # Try to locate the specific section to change
                        line_number = suggested_fix.get("line_number")
                        if line_number and line_number > 0:
                            # Create a modification for a specific section
                            fix_changes.append(
                                CodeChange(
                                    type=ChangeType.MODIFY,
                                    file_path=file_path,
                                    modifications=[
                                        FileModification(
                                            type=ModificationType.REPLACE,
                                            content=fixed_code,
                                            start_line=line_number,
                                            end_line=line_number + original_code.count(
                                                '\n')
                                        )
                                    ]
                                )
                            )
                        else:
                            # Fall back to content replacement
                            fix_changes.append(
                                CodeChange(
                                    type=ChangeType.CREATE,
                                    file_path=file_path,
                                    content=self._replace_substring(
                                        original_change.content,
                                        original_code,
                                        fixed_code
                                    )
                                )
                            )

                elif original_change.type == ChangeType.MODIFY:
                    # For MODIFY changes, we need to adjust the modifications
                    mods = original_change.modifications or []
                    new_mods = []

                    # Check if any modifications match the original code
                    found_match = False
                    for mod in mods:
                        if original_code in mod.content:
                            # Update this modification
                            new_mods.append(
                                FileModification(
                                    type=mod.type,
                                    content=self._replace_substring(
                                        mod.content,
                                        original_code,
                                        fixed_code
                                    ),
                                    start_line=mod.start_line,
                                    end_line=mod.end_line
                                )
                            )
                            found_match = True
                        else:
                            # Keep this modification unchanged
                            new_mods.append(mod)

                    if found_match:
                        fix_changes.append(
                            CodeChange(
                                type=ChangeType.MODIFY,
                                file_path=file_path,
                                modifications=new_mods
                            )
                        )
                    else:
                        # No match found, add a new modification
                        line_number = suggested_fix.get("line_number")
                        if line_number and line_number > 0:
                            all_mods = mods + [
                                FileModification(
                                    type=ModificationType.REPLACE,
                                    content=fixed_code,
                                    start_line=line_number,
                                    end_line=line_number + original_code.count(
                                        '\n')
                                )
                            ]
                            fix_changes.append(
                                CodeChange(
                                    type=ChangeType.MODIFY,
                                    file_path=file_path,
                                    modifications=all_mods
                                )
                            )
            else:
                # File doesn't exist in original changes, might be a new dependency
                if fixed_code:
                    fix_changes.append(
                        CodeChange(
                            type=ChangeType.CREATE,
                            file_path=file_path,
                            content=fixed_code
                        )
                    )

        logger.info(f"Generated {len(fix_changes)} fix changes")
        return fix_changes

    def get_error_history(
            self,
            error_type: Optional[str] = None,
            resolved_only: bool = False
    ) -> List[ErrorRecord]:
        """
        Get error history, optionally filtered.

        Args:
            error_type: Optional error type to filter by
            resolved_only: Whether to include only resolved errors

        Returns:
            List of matching error records
        """
        filtered = self._error_history

        if error_type:
            filtered = [e for e in filtered if e.error_type == error_type]

        if resolved_only:
            filtered = [e for e in filtered if e.resolved]

        return filtered

    def mark_error_resolved(self, error_message: str, resolution: Dict) -> None:
        """
        Mark an error as resolved with the given resolution.

        Args:
            error_message: Error message to mark as resolved
            resolution: Resolution details
        """
        for record in self._error_history:
            if self._is_similar_error(record.error_message, error_message):
                record.resolved = True
                record.resolution_attempt = resolution

    def _find_similar_errors(self, error_message: str) -> List[ErrorRecord]:
        """Find similar errors in the history."""
        return [
            record for record in self._error_history
            if self._is_similar_error(record.error_message, error_message)
        ]

    def _is_similar_error(self, error1: str, error2: str) -> bool:
        """Check if two error messages are similar."""
        # Remove variable parts like line numbers, memory addresses, timestamps
        pattern = r'(0x[0-9a-f]+|line \d+|\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|"\w+\.py")'
        clean1 = re.sub(pattern, 'XXX', error1)
        clean2 = re.sub(pattern, 'XXX', error2)

        # Calculate similarity (simple approach)
        return clean1 == clean2 or clean1 in clean2 or clean2 in clean1

    def _record_error(self, error_message: str, analysis: Dict) -> None:
        """Record an error for future reference."""
        self._error_history.append(
            ErrorRecord(
                error_message=error_message,
                error_type=analysis.get("error_type", "Unknown error"),
                context={
                    "error_location": analysis.get("error_location"),
                    "root_cause": analysis.get("root_cause")
                }
            )
        )

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

    def _is_small_diff(self, original: str, modified: str) -> bool:
        """Check if the difference between two strings is small."""
        import difflib
        matcher = difflib.SequenceMatcher(None, original, modified)
        return matcher.ratio() > 0.9

    def _replace_substring(self, text: str, old: str, new: str) -> str:
        """Replace a substring in text, handling whitespace gracefully."""
        if old in text:
            return text.replace(old, new)

        # Try with normalized whitespace
        old_normalized = ' '.join(old.split())
        text_normalized = ' '.join(text.split())

        if old_normalized in text_normalized:
            # Find position in normalized text
            start = text_normalized.find(old_normalized)
            end = start + len(old_normalized)

            # Replace in normalized text
            result_normalized = text_normalized[:start] + ' '.join(
                new.split()) + text_normalized[end:]

            # Convert back to original format (approximate)
            # This is a best-effort approach and may not preserve exact whitespace
            return result_normalized

        # If no match found, return original text
        return text