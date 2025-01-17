"""
This file is used to implement the RequirementEngineering step of the agent
pipeline. In this step we'll analyse the information given to us by the user,
determine if enough information has been given and then create a requirement
object that can be passed on to following steps.
"""

import os
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from openai import OpenAI

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

    def __init__(
        self,
        prompt_model: Optional[str] = "gpt-4",
        openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
    ) -> None:
        """Initialize the requirements engineering step with an OpenAI client."""
        super().__init__()

        # Initialise large LLM client. At this point, only OpenAI is available.
        self._prompt_model = prompt_model
        if self._prompt_model in ("gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"):
            self._client = OpenAI(api_key=openai_api_key)
        else:
            raise ValueError(f"The model {self._prompt_model} is not supported.")

    def _validate_requirements(self, prompt: str) -> RequirementsSchema:
        """
        Validate the prompt against requirements schema using LLM analysis.

        This method sends the user's prompt to an LLM to analyze and extract
        requirements. The LLM determines which requirements are present,
        missing, or invalid based on the information provided.

        Args:
            prompt: The user's task prompt

        Returns:
            RequirementsSchema with validation results
        """
        analysis_prompt = f"""
        Analyze the following task prompt and extract requirements based on our schema.
        For each requirement, determine if it is present, missing, or invalid (insufficient information).

        Task Prompt: {prompt}

        Requirements to identify:
        1. Task Type (Required) - Must be one of: "design document", "investigation", "implementation"
        2. Description (Required) - Clear description of the task
        3. Definition of Done (Required) - List of acceptance criteria
        4. Risk Level (Required) - Must be one of: "very low", "low", "medium", "high", "very high"
        5. Effort Level (Required) - Must be one of: "very low", "low", "medium", "high", "very high"
        6. Focus Region (Optional) - Specific parts of the codebase to focus on

        For each requirement, respond in the following JSON format:
        {{
            "task_type": {{ "value": "...", "status": "VALID|INVALID|MISSING", "message": "..." }},
            "description": {{ "value": "...", "status": "VALID|INVALID|MISSING", "message": "..." }},
            "dod": {{ "value": ["..."], "status": "VALID|INVALID|MISSING", "message": "..." }},
            "risk": {{ "value": "...", "status": "VALID|INVALID|MISSING", "message": "..." }},
            "effort": {{ "value": "...", "status": "VALID|INVALID|MISSING", "message": "..." }},
            "focus_region": {{ "value": "...", "status": "VALID|INVALID|MISSING", "message": "..." }}
        }}
        
        Please ONLY respond with the JSON format above and nothing else.
        """

        try:
            response = self._client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",
                     "content": "You are a requirements analysis assistant. Your task is to analyze prompts and extract structured requirements."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1  # Low temperature for more consistent analysis
            )

            # Parse the LLM's JSON response
            requirements_data = json.loads(response.choices[0].message.content)

            validations = {}
            try:
                schema = RequirementsSchema(
                    task_type=TaskType(requirements_data["task_type"]["value"])
                    if requirements_data["task_type"][
                           "status"] == "VALID" else None,
                    description=requirements_data["description"]["value"]
                    if requirements_data["description"][
                           "status"] == "VALID" else "",
                    dod=requirements_data["dod"]["value"]
                    if requirements_data["dod"]["status"] == "VALID" else [],
                    risk=RiskLevel(requirements_data["risk"]["value"])
                    if requirements_data["risk"]["status"] == "VALID" else None,
                    effort=EffortLevel(requirements_data["effort"]["value"])
                    if requirements_data["effort"][
                           "status"] == "VALID" else None,
                    focus_region=requirements_data["focus_region"]["value"]
                    if requirements_data["focus_region"][
                           "status"] == "VALID" else None
                )

                # Create validation results for each requirement
                for field, data in requirements_data.items():
                    validations[field] = RequirementValidation(
                        status=RequirementStatus[data["status"]],
                        message=data["message"]
                    )

            except (ValueError, KeyError) as e:
                logger.error(
                    f"Error creating schema from LLM response: {str(e)}")
                raise ValueError("Invalid LLM response format")

            schema.validation_result = validations
            return schema

        except Exception as e:
            logger.error(f"Error during requirements validation: {str(e)}")
            raise ValueError(f"Requirements validation failed: {str(e)}")

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