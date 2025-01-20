"""
This file is used to implement the RequirementEngineering step of the agent
pipeline. In this step we'll analyse the information given to us by the user,
determine if enough information has been given and then create a requirement
object that can be passed on to following steps. We also ensure that this step
integrates with the feedback system, where we will correspond with the user if
feedback is required and use an LLM to handle the validation.
"""

import os
import json
from typing import Dict, Optional, Tuple

from openai import OpenAI

from code_assistant.feedback.manager import FeedbackManager
from code_assistant.feedback.mixins import FeedbackEnabled
from code_assistant.logging.logger import get_logger
from code_assistant.pipeline.step import PipelineStep
from .schema import EffortLevel, RequirementsSchema, RiskLevel, TaskType

logger = get_logger(__name__)


class RequirementsGatherer(PipelineStep, FeedbackEnabled):
    """
    Pipeline step for gathering and validating requirements.
    Uses LLM for analysis, validation, and feedback generation.
    """

    def __init__(
        self,
        feedback_manager: FeedbackManager,
        prompt_model: Optional[str] = "gpt-4",
        openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY"),
    ) -> None:
        PipelineStep.__init__(self)
        FeedbackEnabled.__init__(self, feedback_manager)

        # Initialise large LLM client. At this point, only OpenAI is available.
        if prompt_model in ("gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"):
            self._client = OpenAI(api_key=openai_api_key)
        else:
            raise ValueError(f"The model {prompt_model} is not supported.")

        self._prompt_model = prompt_model

    def _analyze_requirements(
            self, prompt: str, current_schema: Optional[RequirementsSchema] = None
    ) -> Tuple[RequirementsSchema, Optional[str]]:
        """
        Use LLM to analyze requirements and generate feedback if needed.

        Args:
            prompt: The user's task prompt
            current_schema: Optional, existing requirements schema

        Returns:
            Tuple of (updated schema, feedback message if requirements are missing)
        """

        # Construct the analysis prompt
        if current_schema:
            analysis_prompt = f"""
            Analyze the following new information in the context of the current requirements:

            Current Requirements:
            {current_schema.to_markdown()}

            New Information:
            {prompt}
            """
        else:
            analysis_prompt = f"""
            Analyze the following task prompt and extract all requirements into JSON format.
            If any information is missing, provide a clear and specific message asking for it.
            Only include fields where information is clearly provided in the prompt or can be easily inferred.
            If you can reasonably infer any requirements, do so and note your assumptions.

            Task Prompt: {prompt}
            """

        analysis_prompt += """
        Requirements to identify:
        1. Task Type (Required) - Must be one of: "design document", "investigation", "implementation"
        2. Description (Required) - Clear description of the task
        3. Definition of Done (Required) - List of acceptance criteria
        4. Risk Level (Required) - Must be one of: "very low", "low", "medium", "high", "very high"
        5. Effort Level (Required) - Must be one of: "very low", "low", "medium", "high", "very high"
        6. Focus Region (Optional) - Specific parts of the codebase to focus on, this is optional. if not provided, do not ask for it.
        
        Respond with a JSON object only, containing:
        1. requirements: The extracted/updated requirements
        2. feedback_message: A message asking for specific missing information, or null if complete
        3. complete: Boolean indicating if all required information is present
        4. assumptions: List of any assumptions made when inferring requirements

        Example Response:
        {
            "requirements": {
                "task_type": "implementation",
                "description": "...",
                "dod": ["..."],
                "risk": "medium",
                "effort": "low",
                "focus_region": "..."
            },
            "feedback_message": "Could you please specify...",
            "complete": false,
            "assumptions": ["Assumed medium risk due to..."]
        }
        
        Please ONLY respond with the JSON format above and nothing else. If fields are empty,
        they should not be included in the response.
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

            response_content = response.choices[0].message.content
            result = json.loads(response_content)

            schema = RequirementsSchema()
            reqs = result["requirements"]

            if "task_type" in reqs:
                schema.task_type = TaskType(reqs["task_type"])

            if "description" in reqs:
                schema.description = reqs["description"]

            if "dod" in reqs:
                schema.dod = reqs["dod"]

            if "risk" in reqs:
                schema.risk = RiskLevel(reqs["risk"])

            if "effort" in reqs:
                schema.effort = EffortLevel(reqs["effort"])

            if "focus_region" in reqs:
                schema.focus_region = reqs["focus_region"]

            # Log any assumptions made
            if result.get("assumptions"):
                logger.info("Assumptions made during analysis:")
                for assumption in result["assumptions"]:
                    logger.info(f"- {assumption}")

            return schema, result.get("feedback_message")

        except Exception as e:
            logger.error(f"Error during requirements analysis: {str(e)}")
            raise ValueError(f"Requirements analysis failed: {str(e)}")

    def _request_confirmation(self, schema: RequirementsSchema) -> bool:
        """
        Request user confirmation of the final requirements.

        Args:
            schema: The complete requirements schema

        Returns:
            True if user confirms, False if rejected
        """
        response = self.request_step_feedback(
            context="requirements_confirmation",
            prompt=(
                "Please review the final requirements below and confirm if they are correct:\n\n"
                f"{schema.to_markdown()}\n\n"
                "Are these requirements correct? (yes/no)"
            )
        )
        return response.lower().strip() in ('y', 'yes')

    def execute(self, context: Dict) -> None:
        """
        Execute the requirements gathering step with LLM-driven feedback.

        Args:
            context: Pipeline context containing the prompt and other data
        """
        if "prompt" not in context:
            raise ValueError("Prompt not found in pipeline context")

        prompt = context["prompt"]
        logger.info("Starting requirements engineering step")

        # Initial requirements analysis
        schema, feedback_message = self._analyze_requirements(prompt)

        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        console.print("\nInitial Requirements Analysis:")
        console.print("=" * 50)
        console.print(Markdown(schema.to_markdown()))

        # Iteratively collect missing requirements through feedback
        while feedback_message:
            response = self.request_step_feedback(
                context="requirements_gathering",
                prompt=feedback_message
            )

            schema, feedback_message = self._analyze_requirements(
                response, current_schema=schema
            )

            console.print("\nUpdated Requirements Analysis:")
            console.print("=" * 50)
            console.print(Markdown(schema.to_markdown()))

        # Request final confirmation
        while not self._request_confirmation(schema):
            response = self.request_step_feedback(
                context="requirements_update",
                prompt="What would you like to change in these requirements?"
            )

            schema, feedback_message = self._analyze_requirements(
                response, current_schema=schema
            )

            console.print("\nUpdated Requirements Analysis:")
            console.print("=" * 50)
            console.print(Markdown(schema.to_markdown()))

        context["requirements_schema"] = schema

        return self.execute_next(context)