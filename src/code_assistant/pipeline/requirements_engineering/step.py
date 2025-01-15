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
from code_assistant.pipeline.pipeline import PipelineStep

logger = get_logger(__name__)


class RequirementStatus(Enum):
    """Status of a requirement validation."""
    MISSING = auto()
    INVALID = auto()
    VALID = auto()

@dataclass
class RequirementValidation:
    """Result of validating a requirement."""
    status: RequirementStatus
    message: str

@dataclass
class RequirementsSchema:
    """Schema for task requirements."""
    task_type: str
    description: str
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    validation_result: Optional[Dict[str, RequirementValidation]] = field(default=None)

    def is_valid(self) -> bool:
        """Check if all requirements are valid."""
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