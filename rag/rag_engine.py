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
import json
import logging
import os
from typing import List, Optional

import fire
from dotenv import load_dotenv
from openai import OpenAI
from embedding.compare_embeddings import EmbeddingSimilaritySearch, SearchResult
from embedding.generate_embeddings import CodeEmbedder

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(
            self,
            code_units_path: str,
            prompt_model: Optional[str] = "gpt-4",
            embedding_model: Optional[str] = "microsoft/codebert-base",
            top_k: Optional[int] = 5,
            threshold: Optional[float] = None,
            logging_enabled: Optional[bool] = False,
    ):
        """
        Initialises the RAG engine.

        Args:
            code_units_path: Path to JSON file containing pre-embedded code units.
            prompt_model: Name of the large LLM model to use for response
                generation. Defaults to "gpt-4".
            embedding_model: Name of the model to use for generating embeddings.
                Defaults to "microsoft/codebert-base".
            top_k: Maximum number of similar code units to retrieve. Defaults to 5.
            threshold: Minimum similarity score (0-1) required for retrieved code
                units. Defaults to None.
            logging_enabled: Whether to log detailed information about the RAG
                pipeline execution. Defaults to False.

        Raises:
           FileNotFoundError: If the code_units_path file cannot be found.
           ValueError: If an unsupported prompt_model is specified.
        """
        self._prompt_model = prompt_model
        self._top_k = top_k
        self._threshold = threshold
        self._logging_enabled = logging_enabled

        if not os.path.exists(code_units_path):
            raise FileNotFoundError(
                f"Embedded code units file not found: {code_units_path}\n" 
                "Please provide the correct path to the JSON file."
            )

        # Load code units at initialization
        with open(code_units_path, "r", encoding="utf-8") as f:
            self._code_units = json.load(f)

        # Initialize embedder and similarity searcher
        self._embedder = CodeEmbedder(embedding_model=embedding_model)
        self._searcher = EmbeddingSimilaritySearch(self._code_units)

        # Initialise large LLM client. At this point, only OpenAI is available.
        if self._prompt_model in ("gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"):
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"The model {self._prompt_model} is not supported.")

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
        query_vector = self._embedder.generate_embedding(query)

        # Find code units similar to the query
        similar_code_units = self._searcher.find_similar(
            query_vector=query_vector,
            top_k=self._top_k,
            threshold=self._threshold
        )

        if self._logging_enabled:
            logging.info("Similar Code Units:\n")
            for result in similar_code_units:
                logging.info(f"Similarity: {result.similarity_score}")
                logging.info(f"Code unit: {result.code_unit.get('name')}")
                logging.info("---")

        return similar_code_units

    def _augment(
            self,
            query: str,
            similar_code_units: List[SearchResult]
    ) -> str:
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

        for i, code_unit in enumerate(similar_code_units, 1):
            unit = code_unit.code_unit
            prompt += f"[Code Unit {i}] "
            prompt += f"Type: {unit.get('type', 'N/A')}  |  "
            prompt += f"File: {unit.get('filepath', 'N/A')}\n"

            # Add class context if it exists
            if unit.get('class'):
                prompt += f"Class: {unit['class']}\n"

            # Add source code
            prompt += "```python\n"
            prompt += f"{unit.get('source_code', 'No source code available')}\n"
            prompt += "```\n"

            # Add docstring if it exists and isn't empty
            if unit.get('docstring') and unit['docstring'].strip():
                prompt += f"Documentation: {unit['docstring']}\n"

            prompt += f"Similarity Score: {code_unit.similarity_score:.2f}\n\n"

        prompt += "Based on the code context above, please "
        prompt += "provide a detailed answer to the question. "
        prompt += "Reference specific parts of the code where relevant.\n"

        if self._logging_enabled:
            logging.info("\nAugmented Prompt:\n")
            logging.info(prompt)

        return prompt

    def _generate(self, prompt) -> str:
        """
        Generates a response using the augmented prompt via the OpenAI API.

        Args:
            prompt (str): The augmented prompt.

        Returns:
            response_message (str): The generated response from the large LLM.
        """
        response = self._client.chat.completions.create(
            model=self._prompt_model,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant with expertise in Python programming."},
                {"role": "user", "content": prompt}
            ],
        )
        response_message = response.choices[0].message.content

        if self._logging_enabled:
            logging.info("\nGenerated Response:\n")
            logging.info(response_message)

        return response_message


def main(
        query: str,
        code_units_path: str,
        prompt_model: Optional[str] = "gpt-4",
        embedding_model: Optional[str] = "microsoft/codebert-base",
        top_k: Optional[int] = 5,
        threshold: Optional[float] = None,
        logging_enabled: Optional[bool] = True,
):
    """
    Command-line interface for testing and debugging the RAG engine.

    Args:
       query: The question or request to process.
       code_units_path: Path to the JSON file containing pre-embedded code units.
       prompt_model: Name of the LLM model for response generation. Defaults to
           "gpt-4".
       embedding_model: Name of the model for generating embeddings. Defaults to
           "microsoft/codebert-base".
       top_k: Maximum number of similar code units to retrieve. Defaults to 5.
       threshold: Minimum similarity score (0-1) for retrieved code units.
           Defaults to None.
       logging_enabled: Whether to enable detailed logging. Defaults to True.

    Raises:
       FileNotFoundError: If the input_path file cannot be found.

    Example:
       $ python rag_engine.py --query "How do I handle errors?" \
                             --input_path "./code_units.json" \
                             --logging_enabled True
    """
    code_units_path= os.path.abspath(code_units_path)
    if not os.path.exists(code_units_path):
        raise FileNotFoundError(
            f"Embedded code units file not found: {code_units_path}\n"
            "Please provide the correct path to your code units JSON file."
        )

    engine = RAGEngine(
        code_units_path=code_units_path,
        prompt_model=prompt_model,
        embedding_model=embedding_model,
        top_k=top_k,
        threshold=threshold,
        logging_enabled=logging_enabled,
    )
    response = engine.process(query=query)

if __name__ == "__main__":
    fire.Fire(main)