"""
This file is used to implement the RequirementEngineering step of the agent
pipeline. In this step we'll analyse the information given to us by the user,
determine if enough information has been given and then create a requirement
object that can be passed on to following steps.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from code_assistant.logging.logger import get_logger
from code_assistant.pipeline.step import PipelineStep

logger = get_logger(__name__)


class RequirementStatus(Enum):
    """
    Status of a requirement's validation result.

    Used to indicate whether a requirement is:
    - MISSING: The requirement was not found in the input
    - INVALID: The requirement was found but is not sufficient
    - VALID: The requirement was found and is sufficient
    """
    MISSING = auto()
    INVALID = auto()
    VALID = auto()


@dataclass
class RequirementValidation:
    """
    Holds the validation result and explanation message for a single requirement.

    Attributes:
        status: The status of the validation (MISSING, INVALID, or VALID)
        message: A descriptive message explaining why the requirement received this status
    """
    status: RequirementStatus
    message: str


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
    task_type: TaskType
    description: str
    dod: List[str]
    risk: RiskLevel
    effort: EffortLevel
    focus_region: Optional[str] = None
    validation_result: Optional[Dict[str, RequirementValidation]] = field(
        default=None)

    def is_valid(self) -> bool:
        """
        Check if all requirements are valid.

        Returns:
            bool: True if all requirements have passed validation, False otherwise
        """
        if not self.validation_result:
            return False
        return all(v.status == RequirementStatus.VALID
                   for v in self.validation_result.values())

class RequirementsEngineering(PipelineStep):
    """
    This step in the pipeline handles requirements engineering. It validates and
    processes the initial task prompt against the requirement schema.
    """

    def _validate_requirements(self, prompt: str) -> RequirementsSchema:
        """
        Validate the prompt against requirements schema.

        Args:
            prompt: The user's task prompt.

        Returns:
            RequirementsSchema with validation results
        """
        # This is a basic implementation - we'll enhance this with actual
        # prompt analysis and validation logic
        schema = RequirementsSchema(
            task_type="code_generation",  # This would be inferred from prompt
            description=prompt,
        )

        # Perform basic validation
        validations = {}

        # Check for task type
        if not schema.task_type:
            validations["task_type"] = RequirementValidation(
                RequirementStatus.MISSING,
                "Task type could not be determined from prompt"
            )
        else:
            validations["task_type"] = RequirementValidation(
                RequirementStatus.VALID,
                "Task type identified"
            )

        # Check for description
        if not schema.description:
            validations["description"] = RequirementValidation(
                RequirementStatus.MISSING,
                "Task description is missing"
            )
        else:
            validations["description"] = RequirementValidation(
                RequirementStatus.VALID,
                "Description provided"
            )

        schema.validation_result = validations
        return schema

    def execute(self, context: Dict) -> None:
        """
        Execute the requirements engineering step.

        Args:
            context: Pipeline context containing the prompt and other data

        Raises:
            ValueError: If the prompt is missing from context
        """
        if "prompt" not in context:
            raise ValueError("Prompt not found in pipeline context")

        prompt = context["prompt"]
        logger.info("Starting requirements engineering step")

        # Validate requirements
        schema = self._validate_requirements(prompt)

        # Add schema to context
        context["requirements_schema"] = schema

        if not schema.is_valid():
            logger.warning("Requirements validation failed:")
            for key, validation in schema.validation_result.items():
                if validation.status != RequirementStatus.VALID:
                    logger.warning(f"- {key}: {validation.message}")
            raise ValueError("Requirements validation failed")

        logger.info("Requirements engineering step completed successfully")
        return self.execute_next(context)