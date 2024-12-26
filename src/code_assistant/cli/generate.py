import asyncio
from pathlib import Path
from typing import Optional, List

from code_assistant.evaluation.data_generators.prompt_code_pair_generator import (
    OpenAIGenerator,
    OpenAIConfig,
)
from code_assistant.storage.code_store import CodebaseSnapshot


class GenerateCommands:
    """Commands for generating training data."""

    def prompts(
        self,
        code_units_path: str,
        output_path: Optional[str] = None,
        num_rows: Optional[int] = None,
        unit_types: Optional[List[str]] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 150,
    ) -> None:
        """Generate prompt-code pairs using OpenAI."""
        codebase = CodebaseSnapshot.from_json(Path(code_units_path))

        unit_types = unit_types or ["function", "method", "class"]

        if output_path is None:
            output_path = Path(code_units_path).parent / "prompt_code_pairs.json"

        generator = OpenAIGenerator(
            codebase=codebase,
            config=OpenAIConfig(temperature=temperature, max_tokens=max_tokens),
            output_path=Path(output_path),
            num_rows=num_rows,
            unit_types=unit_types,
        )

        asyncio.run(generator.generate_and_save())
