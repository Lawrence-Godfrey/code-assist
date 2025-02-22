import os
from abc import ABC, abstractmethod
from typing import Optional

from code_assistant.logging.logger import get_logger
from code_assistant.models.factory import Model, ModelFactory

logger = get_logger(__name__)


class PromptModel(Model, ABC):
    """
    Abstract base class for all prompt models.

    This class defines the interface that all prompt model implementations
    must follow, ensuring consistent behavior across different providers.
    """

    def __init__(self, model_name: str):
        """
        Initialize the prompt model.

        Args:
            model_name: Name of the specific model to use
        """
        super().__init__(model_name)

    @abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the model for the given prompts.

        Args:
            system_prompt: The system context/instruction for the model
            user_prompt: The specific user query or instruction
            temperature: Controls randomness in the response (0.0 to 1.0)
            max_tokens: Optional maximum length of the response

        Returns:
            The generated response as a string

        Raises:
            ValueError: If the model name is invalid or parameters are incorrect
        """
        pass


@ModelFactory.register("gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "gpt-4", "gpt-3.5")
class OpenAIPromptModel(PromptModel):
    """OpenAI-specific implementation of the prompt model interface."""

    def __init__(self, model_name: str):
        """
        Initialize the OpenAI prompt model.

        Args:
            model_name: Name of the specific OpenAI model to use

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
        """
        from openai import OpenAI

        super().__init__(model_name)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OPENAI_API_KEY environment variable is not set"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self._client = OpenAI(api_key=api_key)

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response using the OpenAI Chat Completions API.

        Args:
            system_prompt: System context/instruction for the model
            user_prompt: User query or instruction
            temperature: Controls response randomness (0.0 to 1.0)
            max_tokens: Optional maximum response length

        Returns:
            Generated response as a string

        Raises:
            ValueError: If temperature or max_tokens is out of valid range
            RuntimeError: If the API call fails
        """
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")

        try:
            # Prepare the API call parameters
            params = {
                "model": self._model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }

            # Add max_tokens if specified
            if max_tokens is not None:
                if max_tokens <= 0:
                    raise ValueError("max_tokens must be greater than 0")
                params["max_tokens"] = max_tokens

            # Make the API call
            response = self._client.chat.completions.create(**params)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise RuntimeError(f"OpenAI API call failed: {str(e)}") from e


@ModelFactory.register("claude-3-5-sonnet-20241022")
class AnthropicPromptModel(PromptModel):
    """Anthropic-specific implementation of the prompt model interface."""

    def __init__(self, model_name: str):
        """
        Initialize the Anthropic prompt model.

        Args:
            model_name: Name of the specific Anthropic model to use

        Raises:
            ValueError: If the model name is not a valid Anthropic model
        """
        from anthropic import Anthropic

        super().__init__(model_name)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            error_msg = "ANTHROPIC_API_KEY environment variable is not set"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self._client = Anthropic(api_key=api_key)

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,  # Max tokens must be set for Anthropic
    ) -> str:
        """
        Generate a response using the Anthropic API.

        Args:
            system_prompt: System context/instruction for the model
            user_prompt: User query or instruction
            temperature: Controls response randomness (0.0 to 1.0)
            max_tokens: Maximum response length

        Returns:
            Generated response as a string

        Raises:
            ValueError: If temperature is out of valid range
            RuntimeError: If the API call fails
        """
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        try:
            # Combine system and user prompts as per Anthropic's format
            combined_prompt = f"{system_prompt}\n\nHuman: {user_prompt}\n\nAssistant:"

            # Prepare the API call parameters
            params = {
                "model": self._model_name,
                "messages": [{"role": "user", "content": combined_prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # Make the API call
            response = self._client.messages.create(**params)

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            raise RuntimeError(f"Anthropic API call failed: {str(e)}") from e


@ModelFactory.register("deepseek-chat", "deepseek-reasoner")
class DeepSeekPromptModel(OpenAIPromptModel):
    """
    DeepSeek-specific implementation of the prompt model interface.
    Inherits from OpenAIPromptModel since DeepSeek uses OpenAI's SDK.
    """

    def __init__(self, model_name: str):
        """
        Initialize the DeepSeek prompt model.

        Args:
            model_name: Name of the specific DeepSeek model to use

        Raises:
            ValueError: If the model name is not a valid DeepSeek model
        """
        # Initialize PromptModel with the model name
        from openai import OpenAI

        PromptModel.__init__(self, model_name)

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            error_msg = "DEEPSEEK_API_KEY environment variable is not set"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize OpenAI client with DeepSeek's base URL
        self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
