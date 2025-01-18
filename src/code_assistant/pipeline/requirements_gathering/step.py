"""
This file is used to implement the RequirementEngineering step of the agent
pipeline. In this step we'll analyse the information given to us by the user,
determine if enough information has been given and then create a requirement
object that can be passed on to following steps.
"""

import os
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from openai import OpenAI

from code_assistant.logging.logger import get_logger
from code_assistant.pipeline.step import PipelineStep

logger = get_logger(__name__)


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
    task_type: Optional[TaskType] = None
    description: str = ""
    dod: List[str] = field(default_factory=list)
    risk: Optional[RiskLevel] = None
    effort: Optional[EffortLevel] = None
    focus_region: Optional[str] = None

    _present_fields: Set[str] = field(default_factory=set)
    _required_fields: Set[str] = field(
        default_factory=lambda: {'task_type', 'description', 'dod', 'risk', 'effort'}
    )

    def add_field(self, field_name: str):
        self._present_fields.add(field_name)

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

class RequirementsGatherer(PipelineStep):
    """
    This step in the pipeline handles requirements gathering. It validates and
    processes the initial task prompt against the requirement schema.
    """

    def __init__(
        self,
        prompt_model: Optional[str] = "gpt-4",
        openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
    ) -> None:
        """Initialize the requirements gathering step with an OpenAI client."""
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

        Raises:
            ValueError: If LLM response is invalid or requirements validation fails
        """
        analysis_prompt = f"""
        Analyze the following task prompt and extract the requirements into JSON format.
        Only include fields where information is clearly provided in the prompt or can be easily inferred.

        Task Prompt: {prompt}

        Requirements to identify:
        1. Task Type (Required) - Must be one of: "design document", "investigation", "implementation"
        2. Description (Required) - Clear description of the task
        3. Definition of Done (Required) - List of acceptance criteria
        4. Risk Level (Required) - Must be one of: "very low", "low", "medium", "high", "very high"
        5. Effort Level (Required) - Must be one of: "very low", "low", "medium", "high", "very high"
        6. Focus Region (Optional) - Specific parts of the codebase to focus on

        Respond in the following JSON format, only including fields that are clearly specified:
        {{
            "task_type": "type_value",
            "description": "description_text",
            "dod": ["criterion1", "criterion2"],
            "risk": "risk_level",
            "effort": "effort_level",
            "focus_region": "region_path"
        }}
        
        Please ONLY respond with the JSON format above and nothing else.
        """

        try:
            response = self._client.chat.completions.create(
                model=self._prompt_model,
                messages=[
                    {"role": "system",
                     "content": "You are a requirements analysis assistant. Your task is to analyze prompts and extract structured requirements."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1  # Low temperature for more consistent analysis
            )

            # Parse the LLM's JSON response
            requirements_data = json.loads(response.choices[0].message.content)

            # Create schema with default values
            schema = RequirementsSchema()

            # Update fields and track presence
            if "task_type" in requirements_data:
                schema.task_type = TaskType(requirements_data["task_type"])
                schema.add_field("task_type")

            if "description" in requirements_data:
                schema.description = requirements_data["description"]
                schema.add_field("description")

            if "dod" in requirements_data:
                schema.dod = requirements_data["dod"]
                schema.add_field("dod")

            if "risk" in requirements_data:
                schema.risk = RiskLevel(requirements_data["risk"])
                schema.add_field("risk")

            if "effort" in requirements_data:
                schema.effort = EffortLevel(requirements_data["effort"])
                schema.add_field("effort")

            if "focus_region" in requirements_data:
                schema.focus_region = requirements_data["focus_region"]
                schema.add_field("focus_region")

            return schema

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from LLM: {str(e)}")
            raise ValueError("Failed to parse LLM response")
        except ValueError as e:
            logger.error(f"Invalid enum value in LLM response: {str(e)}")
            raise ValueError("Invalid requirement value in LLM response")
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
        context["requirements_schema"] = schema

        # Display the markdown
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        markdown_str = schema.to_markdown()
        console.print("\nTask Requirements Analysis:")
        console.print("=" * 50)
        console.print(Markdown(markdown_str))

        return self.execute_next(context)