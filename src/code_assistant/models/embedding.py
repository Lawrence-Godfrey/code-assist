import os
from abc import ABC, abstractmethod

import numpy as np

from code_assistant.logging.logger import get_logger
from code_assistant.models.factory import Model, ModelFactory
from code_assistant.storage.types import EmbeddingUnit

logger = get_logger(__name__)


class EmbeddingModel(Model, ABC):
    """Abstract base class for embedding models."""

    def __init__(self, model_name: str):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the specific model to use
        """
        super().__init__(model_name)

    @abstractmethod
    def generate_embedding(self, text: str) -> EmbeddingUnit:
        """Generate an embedding vector for the given text."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        pass


@ModelFactory.register(
    "microsoft/codebert-base",
    "jinaai/jina-embeddings-v2-base-code",
    "jinaai/jina-embeddings-v3",
)
class TransformersEmbeddingModel(EmbeddingModel):
    """Implementation for Hugging Face Transformers-based embedding models."""

    def __init__(self, model_name: str, max_length: int = 512):
        """
        Initialize the Transformers embedding model.

        Args:
            model_name: Name of the Hugging Face model
            max_length: Maximum token length for input sequences
        """
        import torch
        from transformers import AutoModel, AutoTokenizer

        super().__init__(model_name)

        # Store torch module as instance variable to maintain access
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def generate_embedding(self, text: str) -> EmbeddingUnit:
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
            with self._torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling over the last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Normalize embedding to unit length
            embedding = embeddings.cpu().numpy().flatten()
            return EmbeddingUnit(
                embedding / np.linalg.norm(embedding), self._model_name
            )

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        return self.model.config.hidden_size


@ModelFactory.register(
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
)
class OpenAIEmbeddingModel(EmbeddingModel):
    """Implementation for OpenAI API-based embedding models."""

    # We hardcode the model-embedding-size pairs here instead of sending an
    # extra query to the OpenAI API for model information
    MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str):
        """
        Initialize the OpenAI embedding model.

        Args:
            model_name: Name of the OpenAI embedding model
        """
        from openai import OpenAI

        super().__init__(model_name)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OPENAI_API_KEY environment variable is not set"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.client = OpenAI(api_key=api_key)
        self._embedding_dimension = self.MODELS[model_name]

    def generate_embedding(self, text: str) -> EmbeddingUnit:
        """Generate an embedding vector for the given text."""
        try:
            response = self.client.embeddings.create(
                model=self._model_name, input=text, encoding_format="float"
            )

            # Extract the embedding vector from the response
            embedding = np.array(response.data[0].embedding)

            # OpenAI embeddings are already normalized, but for consistency...
            return EmbeddingUnit(
                embedding / np.linalg.norm(embedding), self._model_name
            )

        except Exception as e:
            logger.error(f"OpenAI embedding generation error: {e}")
            raise

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        return self._embedding_dimension
