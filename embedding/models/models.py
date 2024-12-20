from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from storage.code_store import CodeEmbedding


class EmbeddingModelFactory:
    """Factory class for creating embedding models."""

    MODEL_REGISTRY = {
        "microsoft/codebert-base": "TransformersEmbeddingModel",
        "jinaai/jina-embeddings-v2-base-code": "TransformersEmbeddingModel",
        "jinaai/jina-embeddings-v3": "TransformersEmbeddingModel",
        "text-embedding-3-small": "OpenAIEmbeddingModel",
        "text-embedding-3-large": "OpenAIEmbeddingModel",
        "text-embedding-ada-002": "OpenAIEmbeddingModel",
    }

    @classmethod
    def create(
        cls,
        embedding_model: str,
        max_length: Optional[int] = 512,
        openai_api_key: Optional[str] = None,
    ) -> "EmbeddingModel":
        """
        Create and return an appropriate embedding model instance.

        Args:
            embedding_model: Name of the embedding model to create
            max_length: Maximum token length for input sequences
                       (only applicable to TransformersEmbeddingModel)
            openai_api_key: OpenAI API key (required for OpenAI models)

        Returns:
            An instance of EmbeddingModel

        Raises:
            ValueError: If the specified model is not supported or if OpenAI API
                key is missing for OpenAI models
        """
        if embedding_model not in cls.MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model: {embedding_model}. "
                f"Supported models are: {list(cls.MODEL_REGISTRY.keys())}"
            )

        model_type = cls.MODEL_REGISTRY[embedding_model]

        # Return the appropriate model class
        if model_type == "TransformersEmbeddingModel":
            return TransformersEmbeddingModel(embedding_model, max_length)
        elif model_type == "OpenAIEmbeddingModel":
            if openai_api_key is None:
                raise ValueError(
                    "OpenAI API key is required for OpenAI embedding models."
                )
            return OpenAIEmbeddingModel(embedding_model, openai_api_key)

        raise ValueError(f"Unknown model type: {model_type}")


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    model_name: str

    @abstractmethod
    def generate_embedding(self, text: str) -> CodeEmbedding:
        """Generate an embedding vector for the given text."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        pass


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
        from transformers import AutoTokenizer, AutoModel

        # Store torch module as instance variable to maintain access
        self._torch = torch

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def generate_embedding(self, text: str) -> CodeEmbedding:
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
            return CodeEmbedding(embedding / np.linalg.norm(embedding), self.model_name)

        except Exception as e:
            print(f"Embedding generation error: {e}")
            raise

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        return self.model.config.hidden_size


class OpenAIEmbeddingModel(EmbeddingModel):
    """Implementation for OpenAI API-based embedding models."""

    # We hardcode the model-embedding-size pairs here instead of sending an
    # extra query to the OpenAI API for model information
    MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize the OpenAI embedding model.

        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key
        """
        from openai import OpenAI

        self.model_name = model_name
        self._embedding_dimension = self.MODELS[model_name]
        self.client = OpenAI(api_key=api_key)

    def generate_embedding(self, text: str) -> CodeEmbedding:
        """Generate an embedding vector for the given text."""
        try:
            response = self.client.embeddings.create(
                model=self.model_name, input=text, encoding_format="float"
            )

            # Extract the embedding vector from the response
            embedding = np.array(response.data[0].embedding)

            # OpenAI embeddings are already normalized, but for consistency...
            return CodeEmbedding(embedding / np.linalg.norm(embedding), self.model_name)

        except Exception as e:
            print(f"OpenAI embedding generation error: {e}")
            raise

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        return self._embedding_dimension
