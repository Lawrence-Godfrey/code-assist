import json
import os

import fire
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional

from storage.code_store import CodebaseSnapshot, Method, Class


class CodeEmbedder:
    def __init__(
        self, embedding_model: str = "microsoft/codebert-base", max_length: int = 512
    ):
        """
        Initialize code embedder which generates embeddings for code units and queries.

        Args:
            embedding_model (str): Hugging Face model for generating embeddings
            max_length (int): Maximum token length for input sequences
        """
        # Load tokenizer and embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.max_length = max_length
        self.embedding_dimension = self.model.config.hidden_size

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for text (code or query).

        Args:
            text (str): Text to embed (can be code or natural language query)

        Returns:
            Normalized embedding vector
        """
        try:
            # Tokenize the input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Use mean pooling over the last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Normalize embedding to unit length
            embedding = embeddings.cpu().numpy().flatten()
            return embedding / np.linalg.norm(embedding)

        except Exception as e:
            print(f"Embedding generation error: {e}")
            raise

    def embed_code_units(
        self,
        codebase: CodebaseSnapshot,
    ) -> CodebaseSnapshot:
        """
        Generate embeddings for a list of code units.

        Args:
            codebase (CodebaseSnapshot): Codebase snapshot containing code units

        Returns:
            Updated codebase snapshot with embedded code units.
        """

        for file in codebase:
            for unit in file:
                try:
                    formatted_string = (
                        f"type: {unit.unit_type}, "
                        f"name: {unit.name}, "
                        f"filepath: {unit.file.filepath}, "
                        f"source_code: {unit.source_code}"
                    )
                    unit.embedding = self.generate_embedding(formatted_string)

                    if isinstance(unit, Class):
                        for method in unit.methods:
                            formatted_string = (
                                f"type: {method.unit_type}, "
                                f"filepath: {method.class_ref.file.filepath}, "
                                f"class: {method.class_ref.name}, "
                                f"name: {method.name}, "
                                f"source_code: {method.source_code}"
                            )
                            method.embedding = self.generate_embedding(formatted_string)

                except Exception as e:
                    print(f"Failed to embed unit {unit.name}: {e}")

        return codebase


def process_embeddings(
    input_path: str = "code_units.json",
    output_path: Optional[str] = None,
    model_name: str = "microsoft/codebert-base",
) -> None:
    """
    Generate embeddings for code units from a JSON file.

    Args:
        input_path (str): Path to the JSON file containing code units
                         (defaults to 'code_units.json' in current directory)
        output_path (str, optional): Path to save the embeddings
                                   (defaults to 'embedded_' + input filename)
        model_name (str): Name of the Hugging Face model to use for embeddings
    """
    # Convert input path to absolute path if needed
    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Please provide the correct path to your code units JSON file."
        )

    # Generate default output path if none provided
    if output_path is None:
        input_dir = os.path.dirname(input_path)
        input_filename = os.path.basename(input_path)
        output_path = os.path.join(input_dir, f"embedded_{input_filename}")

    # Load code units
    print(f"Loading code units from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        codebase = CodebaseSnapshot.from_json(json.load(f))

    # Initialize embedder
    embedder = CodeEmbedder(embedding_model=model_name)

    # Generate embeddings
    print("Generating embeddings...")
    codebase_with_embeddings = embedder.embed_code_units(codebase)

    # Save results
    print(f"Saving embeddings to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(codebase_with_embeddings, f, indent=2)

    # Print statistics
    print("\nEmbedding Generation Summary:")
    print(f"Total code units processed: {len(codebase_with_embeddings)}")
    if codebase_with_embeddings:
        print(f"Embedding dimension: {len(codebase_with_embeddings[0]['embedding'])}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(process_embeddings)
