"""
Data models for checkpoint management.

This module defines the core data structures for managing code checkpoints,
providing a structured way to track development progress and enable rollback
capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class CheckpointStatus(Enum):
    """Status of a checkpoint in the development process."""
    CREATED = "created"  # Initial checkpoint creation
    VALIDATED = "validated"  # Passed tests and validation
    REVIEWED = "reviewed"  # Reviewed and approved
    FAILED = "failed"  # Failed tests or validation
    ROLLED_BACK = "rolled_back"  # Checkpoint was rolled back


@dataclass
class TestResult:
    """Results of test execution for a checkpoint."""
    success: bool
    exit_code: int
    output: str
    passed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    coverage: Optional[float] = None
    execution_time: float = 0.0
    error: Optional[str] = None


@dataclass
class Checkpoint:
    """
    Represents a development checkpoint with metadata.

    A checkpoint is a point-in-time snapshot of code changes with
    associated metadata, test results, and status information.
    """
    id: str
    commit_hash: str
    message: str
    branch_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: CheckpointStatus = CheckpointStatus.CREATED
    test_results: Optional[TestResult] = None
    changes: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert the checkpoint to a dictionary for serialization."""
        return {
            "id": self.id,
            "commit_hash": self.commit_hash,
            "message": self.message,
            "branch_name": self.branch_name,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "test_results": {
                "success": self.test_results.success,
                "exit_code": self.test_results.exit_code,
                "output": self.test_results.output,
                "passed_count": self.test_results.passed_count,
                "failed_count": self.test_results.failed_count,
                "error_count": self.test_results.error_count,
                "coverage": self.test_results.coverage,
                "execution_time": self.test_results.execution_time,
                "error": self.test_results.error,
            } if self.test_results else None,
            "changes": self.changes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Checkpoint':
        """Create a Checkpoint from a dictionary."""
        # Handle test results if present
        test_results = None
        if tr_data := data.get("test_results"):
            test_results = TestResult(
                success=tr_data["success"],
                exit_code=tr_data["exit_code"],
                output=tr_data["output"],
                passed_count=tr_data.get("passed_count", 0),
                failed_count=tr_data.get("failed_count", 0),
                error_count=tr_data.get("error_count", 0),
                coverage=tr_data.get("coverage"),
                execution_time=tr_data.get("execution_time", 0.0),
                error=tr_data.get("error"),
            )

        # Create the checkpoint
        return cls(
            id=data["id"],
            commit_hash=data["commit_hash"],
            message=data["message"],
            branch_name=data["branch_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=CheckpointStatus(data["status"]),
            test_results=test_results,
            changes=data.get("changes", []),
            metadata=data.get("metadata", {}),
        )