"""
Models for code changes.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union


class ChangeType(Enum):
    """Types of code changes that can be made."""

    CREATE = "create"  # Create new file
    MODIFY = "modify"  # Modify existing file
    DELETE = "delete"  # Delete file


class ModificationType(Enum):
    """Types of modifications that can be made to existing files."""

    INSERT = "insert"  # Insert at line number
    REPLACE = "replace"  # Replace lines from start to end
    DELETE = "delete"  # Delete lines from start to end


@dataclass
class FileModification:
    """Represents a modification to a specific part of a file."""

    type: ModificationType
    content: str
    start_line: Optional[int] = None  # Line number to start modification
    end_line: Optional[int] = (
        None  # Line number to end modification (for replace/delete)
    )


@dataclass
class CodeChange:
    """Represents a single code change to be applied."""

    type: ChangeType
    file_path: Union[str, Path]  # Relative to workspace root
    content: Optional[str] = None  # Full content for CREATE, none for DELETE
    modifications: Optional[List[FileModification]] = (
        None  # List of modifications for MODIFY
    )


@dataclass
class ChangeResult:
    """Result of applying a single change."""

    success: bool
    file_path: str
    error: Optional[str] = None
