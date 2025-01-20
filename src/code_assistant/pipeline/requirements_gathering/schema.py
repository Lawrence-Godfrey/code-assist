from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

class TaskType(Enum):
    """
    Types of tasks that the agent can perform.

    Values:
        DESIGN_DOCUMENT: Creation of design documentation
        INVESTIGATION: Analysis and research tasks
        IMPLEMENTATION: Code implementation tasks
    """
    DESIGN_DOCUMENT = "design document"
    INVESTIGATION = "investigation"
    IMPLEMENTATION = "implementation"


class RiskLevel(Enum):
    """
    Risk levels for task implementation.

    Represents the potential impact of errors in production:
    - VERY_LOW: Minimal to no impact on system operation
    - LOW: Minor impact, easily fixed
    - MEDIUM: Moderate impact, requires attention
    - HIGH: Significant impact on system operation
    - VERY_HIGH: Critical impact, could cause system failure
    """
    VERY_LOW = "very low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very high"


class EffortLevel(Enum):
    """
    Effort levels for task implementation.

    Similar to story points, indicates the amount of work required:
    - VERY_LOW: Simple changes, minimal coding required
    - LOW: Small changes, straightforward implementation
    - MEDIUM: Moderate changes, some complexity
    - HIGH: Significant changes, complex implementation
    - VERY_HIGH: Major changes, highly complex implementation
    """
    VERY_LOW = "very low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very high"


@dataclass
class RequirementsSchema:
    """
    Schema for task requirements, defining all necessary information for task execution.

    Required Fields:
        task_type: The type of task to be performed (design, investigation, implementation)
        description: Detailed description of what needs to be done
        dod: Definition of Done - list of acceptance criteria
        risk: Assessment of potential production impact
        effort: Estimated work effort required

    Optional Fields:
        focus_region: Specific parts of the codebase to focus on

    Internal Fields:
        validation_result: Results of requirements validation checks

    Examples:
        A typical implementation task might have:
        - task_type: TaskType.IMPLEMENTATION
        - description: "Implement error handling in the pipeline module"
        - dod: ["Add try-catch blocks", "Log errors appropriately", "Add tests"]
        - risk: RiskLevel.MEDIUM
        - effort: EffortLevel.LOW
        - focus_region: "pipeline/error_handling.py"
    """
    _task_type: Optional[TaskType] = field(default=None, init=False)
    _description: str = field(default="", init=False)
    _dod: List[str] = field(default_factory=list, init=False)
    _risk: Optional[RiskLevel] = field(default=None, init=False)
    _effort: Optional[EffortLevel] = field(default=None, init=False)
    _focus_region: Optional[str] = field(default=None, init=False)

    _present_fields: Set[str] = field(default_factory=set)
    _required_fields: Set[str] = field(
        default_factory=lambda: {'task_type', 'description', 'dod', 'risk',
                                 'effort'}
    )

    @property
    def task_type(self) -> Optional[TaskType]:
        return self._task_type

    @task_type.setter
    def task_type(self, value: Optional[TaskType]):
        self._task_type = value
        if value is not None:
            self._present_fields.add('task_type')

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str):
        self._description = value
        if value:
            self._present_fields.add('description')

    @property
    def dod(self) -> List[str]:
        return self._dod

    @dod.setter
    def dod(self, value: List[str]):
        self._dod = value
        if value:
            self._present_fields.add('dod')

    @property
    def risk(self) -> Optional[RiskLevel]:
        return self._risk

    @risk.setter
    def risk(self, value: Optional[RiskLevel]):
        self._risk = value
        if value is not None:
            self._present_fields.add('risk')

    @property
    def effort(self) -> Optional[EffortLevel]:
        return self._effort

    @effort.setter
    def effort(self, value: Optional[EffortLevel]):
        self._effort = value
        if value is not None:
            self._present_fields.add('effort')

    @property
    def focus_region(self) -> Optional[str]:
        return self._focus_region

    @focus_region.setter
    def focus_region(self, value: Optional[str]):
        self._focus_region = value
        if value is not None:
            self._present_fields.add('focus_region')

    def is_valid(self) -> bool:
        """Check if all required fields are present."""
        return all(field in self._present_fields for field in self._required_fields)

    def to_markdown(self, include_missing: bool = True) -> str:
        """
        Convert the requirements schema to a markdown formatted string.

        Args:
            include_missing: Whether to include requirements that are missing or invalid

        Returns:
            Markdown formatted string representing the requirements
        """
        lines = ["# Task Requirements\n"]

        if 'task_type' in self._present_fields:
            lines.extend(["### Task Type", f"{self.task_type.value}\n"])

        if 'description' in self._present_fields:
            lines.extend(["### Description", f"{self.description}\n"])

        if 'dod' in self._present_fields:
            lines.extend(["### Definition of Done"])
            lines.extend(f"- {item}" for item in self.dod)
            lines.append("")

        if 'risk' in self._present_fields:
            lines.extend(["### Risk Level", f"{self.risk.value}\n"])

        if 'effort' in self._present_fields:
            lines.extend(["### Effort Level", f"{self.effort.value}\n"])

        if 'focus_region' in self._present_fields:
            lines.extend(["### Focus Region", f"{self.focus_region}\n"])

        return "\n".join(lines)