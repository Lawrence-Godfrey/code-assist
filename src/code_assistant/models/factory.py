from abc import ABC
from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T", bound="Model")


class ModelFactory:
    """
    Factory for creating model instances.

    This factory manages the registration and creation of model classes for
    embedding and prompt models, ensuring that model names are mapped to their
    correct implementations.
    """

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
        Create and return an appropriate model instance.

        Args:
            model_name: Name of the model to create
            *args: Variable length argument list to pass to the model constructor
            **kwargs: Arbitrary keyword arguments to pass to the model constructor

        Returns:
            An instance of a model class

        Raises:
            ValueError: If model_name is not provided or model is not supported
        """
        if model_name not in cls._models:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models are: {list(cls._models.keys())}"
            )
        return cls._models[model_name](model_name, *args, **kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """Get a list of all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def models(cls) -> Dict[str, Type[T]]:
        """Get a list of all registered model classes."""
        return cls._models

    @classmethod
    def get_default_prompt_model(cls):
        """Get the default prompt model name."""
        return "gpt-4"

    @classmethod
    def get_default_embedding_model(cls):
        """Get the default embedding model name."""
        return "jinaai/jina-embeddings-v3"


class Model(ABC):
    """
    Base class for all models in the code assistant.

    This class provides the common model_name attribute that all models
    must have, whether they are prompt models or embedding models.
    """

    def __init__(self, model_name: str):
        """
        Initialize a model with its name.

        Args:
            model_name: Name/identifier of the model
        """
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """Get the name of the model being used."""
        return self._model_name
