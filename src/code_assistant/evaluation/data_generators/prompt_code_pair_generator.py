import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from code_assistant.storage.code_store import CodebaseSnapshot, CodeUnit
from code_assistant.evaluation.data_generators.prompt_code_pair_dataset import (
    PromptCodePair,
    PromptCodePairDataset,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BaseGeneratorConfig:
    """Base configuration for prompt generation."""

    temperature: float = 0.7
    max_tokens: int = 150
    model: str = field(default="")
    system_prompt: str = """
    You are an expert at understanding code and generating natural language 
    prompts. Generate a clear, specific prompt that would lead a developer or AI 
    to write this exact code. Focus on the functionality and requirements, not the 
    implementation details. The prompt should be specific enough that the code 
    would be a natural solution to it.
    """


@dataclass
class OpenAIConfig(BaseGeneratorConfig):
    """Configuration specific to OpenAI models."""

    model: str = "gpt-4"


class AbstractPromptGenerator(ABC):
    """Abstract base class for prompt-code pair generators."""

    def __init__(
        self,
        codebase: CodebaseSnapshot,
        output_path: Path = Path(
            os.path.expanduser(
                "~/code_assist/datasets/synthetic/prompt_code_pairs.json"
            )
        ),
        num_rows: Optional[int] = None,
        unit_types: List[str] = None,
    ):
        """
        Initialize the prompt generator.

        Args:
            codebase: Snapshot of the codebase to generate prompts for
            output_path: Path to save the generated dataset
            num_rows: Optional limit on number of rows to generate
            unit_types: Types of code units to process (defaults to function, method, class)
        """
        self.codebase = codebase
        self.output_path = output_path
        self.num_rows = num_rows
        self.unit_types = unit_types or ["function", "method", "class"]

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def _generate_prompt(self, code_unit: CodeUnit) -> str:
        """Generate a prompt for a given code unit using the specific model."""
        pass

    def _filter_code_units(self) -> List[CodeUnit]:
        """Filter and limit code units based on configuration."""
        filtered_units = self.codebase.get_units_by_type(self.unit_types)

        if self.num_rows is not None:
            if self.num_rows < len(filtered_units):
                filtered_units = random.sample(filtered_units, self.num_rows)

        return filtered_units

    async def generate_dataset(self) -> PromptCodePairDataset:
        """Generate prompt-code pairs for filtered code units."""
        filtered_units = self._filter_code_units()
        dataset = PromptCodePairDataset()

        for code_unit in tqdm(
            filtered_units, desc=f"Generating prompts with {self.__class__.__name__}"
        ):
            try:
                prompt = await self._generate_prompt(code_unit)
                pair = PromptCodePair(
                    prompt=prompt,
                    code_unit=code_unit,
                )
                dataset.add_pair(pair)
            except Exception as e:
                logging.info(f"Error generating prompt for {code_unit.name}: {str(e)}")
                continue

        return dataset

    async def generate_and_save(self) -> PromptCodePairDataset:
        """Generate prompt-code pairs and save to JSON file."""
        dataset = await self.generate_dataset()
        dataset.to_json(self.output_path)

        return dataset


class OpenAIGenerator(AbstractPromptGenerator):
    """Prompt generator using OpenAI models."""

    def __init__(
        self,
        codebase: CodebaseSnapshot,
        config: OpenAIConfig = None,
        output_path: Path = Path(
            os.path.expanduser(
                "~/code_assist/datasets/synthetic/prompt_code_pairs.json"
            )
        ),
        num_rows: Optional[int] = None,
        unit_types: List[str] = None,
    ):
        """Initialize the OpenAI prompt generator.

        Args:
            codebase: Snapshot of the codebase to generate prompts for
            config: OpenAI configuration (defaults to default OpenAIConfig)
            output_path: Path to save the generated dataset
            num_rows: Optional limit on number of rows to generate
            unit_types: Types of code units to process
        """
        super().__init__(codebase, output_path, num_rows, unit_types)
        self.config = config or OpenAIConfig()
        self._client = AsyncOpenAI()

    async def _generate_prompt(self, code_unit: CodeUnit) -> str:
        """Generate a prompt using OpenAI's chat completion."""
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": f"Generate a prompt for this code:\n\n{code_unit.source_code}",
                },
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content.strip()
