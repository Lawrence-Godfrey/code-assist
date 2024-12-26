import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fire
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from storage.code_store import CodebaseSnapshot, CodeUnit
from evaluation.data_generators.prompt_code_pair_dataset import (
    PromptCodePair,
    PromptCodePairDataset
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
            unit_types: List[str] = None
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
                filtered_units,
                desc=f"Generating prompts with {self.__class__.__name__}"
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
            openai_api_key: str,
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
            openai_api_key: OpenAI API key
            config: OpenAI configuration (defaults to default OpenAIConfig)
            output_path: Path to save the generated dataset
            num_rows: Optional limit on number of rows to generate
            unit_types: Types of code units to process
        """
        super().__init__(codebase, output_path, num_rows, unit_types)
        self.config = config or OpenAIConfig()
        self._client = AsyncOpenAI(api_key=openai_api_key)

    async def _generate_prompt(self, code_unit: CodeUnit) -> str:
        """Generate a prompt using OpenAI's chat completion."""
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": f"Generate a prompt for this code:\n\n{code_unit.source_code}"
                }
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content.strip()


async def main(
    code_units_path: str,
    openai_api_key: str,
    dataset_output_path: Optional[str] = Path(
        os.path.expanduser("~/code_assist/datasets/synthetic/prompt_code_pairs.json")
    ),
    num_rows: Optional[int] = None,
    unit_types: Optional[List[str]] = ["function", "method", "class"],
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = 150,
) -> None:
    """
    Generate prompt-code pairs using OpenAI and save them to a dataset.
    
    Args:
        code_units_path: Path to the JSON file containing code units from a codebase.
        openai_api_key: OpenAI API key for model access.
        dataset_output_path: Path where the generated prompt-code pairs dataset
            will be saved.
        num_rows: Number of code units to process. If None, processes all available units.
        unit_types: List of code unit types to include. Defaults to ["function", "method", "class"].
        temperature: Sampling temperature for the OpenAI model. Higher values makes
            output more random. Defaults to 0.7.
        max_tokens: Maximum number of tokens in generated prompts. Defaults to 150.

    Returns:
        None. Saves the generated dataset to the specified output path.
    """
    # Load codebase from provided path
    codebase = CodebaseSnapshot.from_json(Path(code_units_path))

    # Create generator with provided configuration
    openai_generator = OpenAIGenerator(
        codebase=codebase,
        openai_api_key=openai_api_key,
        output_path=Path(dataset_output_path),
        num_rows=num_rows,
        unit_types=unit_types,
        config=OpenAIConfig(
            temperature=temperature,
            max_tokens=max_tokens
        )
    )

    # Generate dataset and save to specified path
    await openai_generator.generate_and_save()


def run_main(**kwargs):
    """
    Wrapper function to run the async main function with Fire.

    Allows Fire to handle command-line arguments while properly
    managing the async execution.
    """
    return asyncio.run(main(**kwargs))


if __name__ == "__main__":
    fire.Fire(run_main)
