"""
This module implements a RAG (Retrieval Augmented Generation) system for code
assistance on a specific codebase. It combines embedding-based similarity search
with large language models to provide contextually relevant answers to queries.

The system works in three stages:
1. Retrieval: Finds relevant code units by computing similarity between the
   query and code embeddings
2. Augmentation: Enhances the original query with retrieved code context
3. Generation: Uses a large language model to generate detailed responses based
   on the augmented prompt

Usage:
    python rag_engine.py --query "How do I implement X?" \
                        --input_path "path/to/code_units.json" \
                        --model "gpt-4" \
                        --top_k 5 \
                        --threshold 0.5 \
                        --logging_enabled True

Requirements:
    - OpenAI API key in .env file
    - Pre-generated code embeddings in JSON format
"""

import os
from typing import List, Optional

from openai import OpenAI

from code_assistant.embedding.compare_embeddings import (
    EmbeddingSimilaritySearch,
    SearchResult,
)
from code_assistant.embedding.generate_embeddings import CodeEmbedder
from code_assistant.embedding.models.models import EmbeddingModel
from code_assistant.logging.logger import get_logger
from code_assistant.prompt.models import PromptModel
from code_assistant.storage.code_store import CodebaseSnapshot

logger = get_logger(__name__)


class RAGEngine:
    def __init__(
        self,
        codebase: CodebaseSnapshot,
        embedding_model: EmbeddingModel,
        prompt_model: PromptModel,
        top_k: Optional[int] = 5,
        threshold: Optional[float] = None,
    ):
        """
        Initialises the RAG engine.

        Args:
            codebase: CodebaseSnapshot object containing code units and their embeddings.
            embedding_model: EmbeddingModel class used for generating embeddings.
            prompt_model: PromptModel class used for generating prompts.
            top_k: Maximum number of similar code units to retrieve. Defaults to 5.
            threshold: Minimum similarity score (0-1) required for retrieved code
                units. Defaults to None.

        Raises:
           FileNotFoundError: If the code_units_path file cannot be found.
           ValueError: If an unsupported prompt_model is specified.
        """
        self._prompt_model = prompt_model
        self._top_k = top_k
        self._threshold = threshold

        # Initialize embedder and similarity searcher
        self._embedder = CodeEmbedder(embedding_model=embedding_model)
        self._searcher = EmbeddingSimilaritySearch(
            codebase=codebase, embedding_model=embedding_model
        )

    def process(self, query: str) -> str:
        """
        Processes a query through the complete RAG pipeline.

        This method executes the full retrieval-augmentation-generation sequence:
        1. Retrieves similar code units based on the query
        2. Augments the query with the retrieved code context
        3. Generates a response using the specified LLM

        Args:
            query: The user's prompt to which RAG will be applied

        Returns:
            response (str): The generated response that answers the query using
                relevant code context.
        """
        similar_code_units = self._retrieve(query)
        prompt = self._augment(query, similar_code_units)
        response = self._generate(prompt)

        return response

    def _retrieve(self, query: str) -> List[SearchResult]:
        """
        Retrieves relevant code units by computing similarity with the query.

        Args:
            query (str): The user's prompt which will be used to find similar
                code units.

        Returns:
            similar_code_units (List[SearchResult]): The k most similar code
                units to the query, in order from most similar to least similar.
        """
        query_embedding = self._embedder.model.generate_embedding(query)

        # Find code units similar to the query
        similar_code_units = self._searcher.find_similar(
            query_embedding=query_embedding,
            top_k=self._top_k,
            threshold=self._threshold,
        )

        return similar_code_units

    def _augment(self, query: str, similar_code_units: List[SearchResult]) -> str:
        """
        Augments the original query with relevant code context.

        Format of the constructed prompt:
           Question: <original query>

           Here is the relevant code context from the codebase:

           [Code Unit 1] Type: <type> | File: <filepath>
           Class: <class_name>  # if applicable
           ```python
           <source_code>
           ```
           Documentation: <docstring>  # if available
           Similarity Score: <score>

           [Additional code units...]

           Based on the code context above, please provide a detailed answer...

        Args:
            query (str): The user's prompt which will be augmented with similar
                code units for context
            similar_code_units List[SearchResult]: The k most similar code units
                to the query, in order from most similar to least similar.

        Returns:
            prompt (str): The augmented prompt.
        """
        prompt = f"Question: {query}\n\n"
        prompt += "Here is the relevant code context from the codebase:\n\n"

        for i, result in enumerate(similar_code_units, 1):
            unit = result.code_unit
            prompt += f"[Code Unit {i}] "
            prompt += f"Type: {unit.unit_type}  |  "
            prompt += f"Full Name: {unit.fully_qualified_name()}\n"

            # Add source code
            prompt += "```python\n"
            prompt += f"{unit.source_code}\n"
            prompt += "```\n"

            # Add docstring if it exists and isn't empty
            if unit.docstring.strip():
                prompt += f"Documentation: {unit.docstring}\n"

            prompt += f"Similarity Score: {result.similarity_score:.2f}\n\n"

        prompt += "Based on the code context above, please "
        prompt += "provide a detailed answer to the question. "
        prompt += "Reference specific parts of the code where relevant.\n"

        logger.info("\nAugmented Prompt:\n")
        logger.info(prompt)

        return prompt

    def _generate(self, prompt) -> str:
        """
        Generates a response using the augmented prompt via the specified prompt
        model.

        Args:
            prompt (str): The augmented prompt.

        Returns:
            response_message (str): The generated response from the large LLM.
        """
        system_prompt = "You are a helpful assistant with expertise in Python programming."
        response = self._prompt_model.generate_response(
            system_prompt=system_prompt,
            user_prompt=prompt,
        )

        logger.info("\nGenerated Response:\n")
        logger.info(response)

        return response
