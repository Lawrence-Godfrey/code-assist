"""
"""
import json
import os
from typing import List, Optional

import fire
from dotenv import load_dotenv
from openai import OpenAI
from embedding.compare_embeddings import EmbeddingSimilaritySearch, SearchResult
from embedding.generate_embeddings import CodeEmbedder

load_dotenv()

class RagEngine:
    """
    """
    def __init__(
            self,
            input_path: str,
            model: Optional[str] = "gpt-4",
            top_k: Optional[int] = 5,
            threshold: Optional[float] = None,
    ):
        self._input_path = input_path
        self._model = model
        self._top_k = top_k
        self._threshold = threshold
        self._query = ""
        self._prompt = ""
        self._response = ""
        self._similar_code_units = []

        if not os.path.exists(self._input_path):
            raise FileNotFoundError(
                f"Input file not found: {self._input_path}\n"
                "Please provide the correct path to your code units JSON file."
            )

        # Load code units at initialization
        with open(self._input_path, "r", encoding="utf-8") as f:
            self._code_units = json.load(f)

        # Initialize embedder and searcher
        self._embedder = CodeEmbedder()
        self._searcher = EmbeddingSimilaritySearch(self._code_units)

        # Initialise large LLM client
        if self._model in ("gpt-4", "gpt-4o, gpt-4o-mini, gpt-4-turbo"):
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError("That model does not exist")

    def process(self, query: str) -> str:
        self._query = query
        self._retriever()
        self._augmentor()
        self._generator()

        return self._response


    def _retriever(self) -> None:
        query_vector = self._embedder.generate_embedding(self._query)

        # Find code units similar to the query
        self._similar_code_units = self._searcher.find_similar(
            query_vector=query_vector,
            top_k=self._top_k,
            threshold=self._threshold
        )

    def _augmentor(self) -> None:
        self._prompt = f"Question: {self._query}\n\n"
        self._prompt += "Here is the relevant code context from the codebase:\n\n"

        for i, code_unit in enumerate(self._similar_code_units):
            unit = code_unit.code_unit
            self._prompt += f"[Code Unit {i}] "
            self._prompt += f"Type: {unit.get('type', 'N/A')}  |  "
            self._prompt += f"File: {unit.get('filepath', 'N/A')}\n"

            # Add class context if it exists
            if unit.get('class'):
                self._prompt += f"Class: {unit['class']}\n"

            # Add source code
            self._prompt += "```python\n"
            self._prompt += f"{unit.get('source_code', 'No source code available')}\n"
            self._prompt += "```\n"

            # Add docstring if it exists and isn't empty
            if unit.get('docstring') and unit['docstring'].strip():
                self._prompt += f"Documentation: {unit['docstring']}"

            self._prompt += f"Similarity Score: {code_unit.similarity_score:.2f}\n\n"

        self._prompt += "Based on the code context above, please "
        self._prompt += "provide a detailed answer to the question. "
        self._prompt += "Reference specific parts of the code where relevant.\n"

    def _generator(self) -> None:
        response = self._client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant with expertise in Python programming."},
                {"role": "user", "content": self._prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        self._response = response.choices[0].message.content


def main(
        query: str,
        input_path: str,
        model: Optional[str] = "gpt-4",
        top_k: Optional[int] = 5,
        threshold: Optional[float] = None,
):
    """
    """
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Please provide the correct path to your code units JSON file."
        )

    engine = RagEngine(
        input_path=input_path,
        model=model,
        top_k=top_k,
        threshold=threshold,
    )
    response = engine.process(query=query)

if __name__ == "__main__":
    fire.Fire(main)