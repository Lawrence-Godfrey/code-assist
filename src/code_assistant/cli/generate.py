import asyncio
import os
from typing import List, Optional

from code_assistant.evaluation.data_generators.prompt_code_pair_generator import (
    OpenAIConfig,
    OpenAIGenerator,
)
from code_assistant.storage.stores.code import MongoDBCodeStore


class GenerateCommands:
    """Commands for generating training data."""

    def prompts(
        self,
        codebase: str,
        openai_api_key: str,
        database_url: str = "mongodb://localhost:27017/",
        num_rows: Optional[int] = None,
        unit_types: Optional[List[str]] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 150,
    ) -> None:
        """Generate prompt-code pairs using OpenAI."""

        database_url = os.getenv("MONGODB_URL") or database_url

        code_store = MongoDBCodeStore(codebase=codebase, connection_string=database_url)

        unit_types = unit_types or ["function", "method", "class"]

        generator = OpenAIGenerator(
            code_store=code_store,
            openai_api_key=openai_api_key,
            config=OpenAIConfig(temperature=temperature, max_tokens=max_tokens),
            num_rows=num_rows,
            unit_types=unit_types,
        )

        asyncio.run(generator.generate_and_save())
