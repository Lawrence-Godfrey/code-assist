from abc import ABC, abstractmethod

import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

load_dotenv()

class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    model_name: str

    @abstractmethod
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding vector for the given text."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        pass


class TransformersEmbeddingModel(BaseEmbeddingModel):
    """Implementation for Hugging Face Transformers-based embedding models."""

    def __init__(self, model_name: str, max_length: int = 512):
        """
        Initialize the Transformers embedding model.

        Args:
            model_name: Name of the Hugging Face model
            max_length: Maximum token length for input sequences
        """
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding vector for the given text."""
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

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        return self.model.config.hidden_size


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """Implementation for OpenAI API-based embedding models."""

    # We hardcode the model-embedding-size pairs here instead of sending an
    # extra query to the OpenAI API for model information
    MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }

    def __init__(self, model_name: str):
        """
        Initialize the OpenAI embedding model.

        Args:
            model_name: Name of the OpenAI embedding model
        """
        self.model_name = model_name
        self._embedding_dimension = self.MODELS[model_name]
        self.client = OpenAI()

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding vector for the given text."""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                encoding_format="float"
            )

            # Extract the embedding vector from the response
            embedding = np.array(response.data[0].embedding)

            # OpenAI embeddings are already normalized, but for consistency...
            return embedding / np.linalg.norm(embedding)

        except Exception as e:
            print(f"OpenAI embedding generation error: {e}")
            raise

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        return self._embedding_dimension
