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
            query: str,
            input_path: Optional[str] =  "code_units.json",
            output_path: Optional[str] = None
    ):
        self._query = query
        self._input_path = input_path
        self._output_path = output_path

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

        self._similar_code_units = []
        self._augmented_prompt = ""
        self._response = ""

        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    def retriever(self, top_k: int = 5, threshold: Optional[float] = None) -> List[SearchResult]:
        query_vector = self._embedder.generate_embedding(self._query)

        # Find similar code units
        self._similar_code_units = self._searcher.find_similar(
            query_vector=query_vector,
            top_k=top_k,
            threshold=threshold
        )

        if self._output_path:
            with open(self._output_path, "w", encoding="utf-8") as f:
                json_results = [
                    {
                        "unit": code_unit.code_unit,
                        "similarity": code_unit.similarity_score,
                    }
                    for code_unit in self._similar_code_units
                ]
                json.dump(json_results, f, indent=2)

        return self._similar_code_units

    def augmentor(self) -> str:
        self._augmented_prompt = f"Question: {self._query}\n\n"
        self._augmented_prompt += "Here is the relevant code context from the codebase:\n\n"

        for i, code_unit in enumerate(self._similar_code_units):
            unit = code_unit.code_unit
            self._augmented_prompt += f"[Code Unit {i}] "
            self._augmented_prompt += f"Type: {unit.get('type', 'N/A')}  |  "
            self._augmented_prompt += f"File: {unit.get('filepath', 'N/A')}\n"

            # Add class context if it exists
            if unit.get('class'):
                self._augmented_prompt += f"Class: {unit['class']}\n"

            # Add source code
            self._augmented_prompt += "```python\n"
            self._augmented_prompt += f"{unit.get('source_code', 'No source code available')}\n"
            self._augmented_prompt += "```\n"

            # Add docstring if it exists and isn't empty
            if unit.get('docstring') and unit['docstring'].strip():
                self._augmented_prompt += f"Documentation: {unit['docstring']}"

            self._augmented_prompt += f"Similarity Score: {code_unit.similarity_score:.2f}\n\n"

        self._augmented_prompt += "Based on the code context above, please "
        self._augmented_prompt += "provide a detailed answer to the question. "
        self._augmented_prompt += "Reference specific parts of the code where relevant.\n"

        return self._augmented_prompt

    def generator(self) -> str:
        self._response = self._client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant with expertise in Python programming."},
                {"role": "user", "content": self._augmented_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return self._response.choices[0].message.content


def main(query: str, input_path: str = "code_units.json", output_path: Optional[str] = None):
    """
    Example usage of the RAG engine
    """
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Please provide the correct path to your code units JSON file."
        )

    if output_path is None:
        input_dir = os.path.dirname(input_path)
        input_filename = os.path.basename(input_path)
        output_path = os.path.join(input_dir, f"similarities_{input_filename}")

    engine = RagEngine(
        query=query,
        input_path=input_path,
        output_path=output_path,
    )
    similar_code = engine.retriever(top_k=3)
    for result in similar_code:
        print(f"Similarity: {result.similarity_score}")
        print(f"Code unit: {result.code_unit.get('name')}")
        print("---")

    augmented_prompt = engine.augmentor()
    print("Augmented Prompt:")
    print(augmented_prompt)

    print("\nGenerated Response:")
    response = engine.generator()
    print(response)

if __name__ == "__main__":
    fire.Fire(main)