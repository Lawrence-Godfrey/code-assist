from abc import ABC, abstractmethod
from typing import Callable, Type, TypeVar, Dict, Any, Optional, List

import numpy as np

from storage.code_store import CodeEmbedding


T = TypeVar("T", bound="EmbeddingModel")


class EmbeddingModelFactory:
    """Factory class for creating embedding models."""

    _models: Dict[str, Type[T]] = {}

    @classmethod
    def register(cls, *names: str) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a model class with its supported model names.

        Args:
            *names: Variable number of model names that this class supports

        Returns:
            Decorator function that registers the model class
        """

        def decorator(model_class: Type[T]) -> Type[T]:
            for name in names:
                cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def model(cls, model_name: str) -> Optional[Type[T]]:
        """
        Get the registered model class for a given model name.

        Args:
            model_name: Name of the model to look up

        Returns:
            The model class if registered, None otherwise
        """
        return cls._models.get(model_name)

    @classmethod
    def create(cls, model_name: str, *args: Any, **kwargs: Any) -> T:
        """
        Create and return an appropriate embedding model instance.

        Args:
            model_name: Name of the model to create
            *args: Variable length argument list to pass to the model constructor
            **kwargs: Arbitrary keyword arguments to pass to the model constructor

        Returns:
            An instance of EmbeddingModel

        Raises:
            ValueError: If model_name is not provided or model is not supported

        Example:
            model = EmbeddingModelFactory.create(model_name="microsoft/codebert-base", max_length=512)
        """
        if model_name not in cls._models:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models are: {list(cls._models.keys())}"
            )
        return cls._models[model_name](*args, **kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """Get a list of all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def models(cls) -> Dict[str, Type[T]]:
        """Get a list of all registered model classes."""
        return cls._models


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


@EmbeddingModelFactory.register(
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


@EmbeddingModelFactory.register(
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
