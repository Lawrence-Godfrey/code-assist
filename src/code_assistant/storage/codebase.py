import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union

import numpy as np

from code_assistant.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CodeEmbedding:
    """
    Represents an embedding vector along with metadata about how it was generated.
    This allows tracking which model was used and with what parameters.
    """

    vector: np.ndarray
    model_name: str

    def __init__(self, vector: Union[List[float], np.ndarray], model_name: str):
        """
        Initialize a CodeEmbedding from either a list or numpy array.

        Args:
            vector: The embedding vector as a list or numpy array
            model_name: The name of the embedding model used
        """
        if isinstance(vector, list):
            self.vector = np.array(vector)
        elif isinstance(vector, np.ndarray):
            self.vector = vector
        else:
            raise ValueError("Invalid type for embedding vector")

        self.model_name = model_name

    def __list__(self) -> List[float]:
        """Convert the embedding vector to a list."""
        return self.vector.tolist()

    def __len__(self) -> int:
        """Get the length of the embedding vector."""
        return len(self.vector)

    def __eq__(self, other):
        if not isinstance(other, CodeEmbedding):
            return False
        return (
            np.array_equal(self.vector, other.vector)
            and self.model_name == other.model_name
        )

    def to_dict(self) -> dict:
        """Convert the code embedding to a dictionary."""
        return {"vector": self.vector.tolist(), "model_name": self.model_name}

    @classmethod
    def from_dict(cls, data: dict) -> Optional["CodeEmbedding"]:
        """Create a CodeEmbedding from a dictionary."""
        if data is None:
            return None
        return cls(vector=np.array(data["vector"]), model_name=data["model_name"])


@dataclass
class CodeUnit(ABC):
    """
    Abstract base class for all code units (classes, methods, functions).
    Cannot be instantiated directly - must use a concrete subclass.
    """

    name: str
    source_code: str
    docstring: Optional[str] = None
    unit_type: str = field(init=False)
    embeddings: Dict[str, CodeEmbedding] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        """Convert the code unit to a dictionary."""
        output = {
            "id": self.id,
            "unit_type": self.unit_type,
            "name": self.name,
            "source_code": self.source_code,
            "docstring": self.docstring,
            "embeddings": {
                model_name: embedding.to_dict()
                for model_name, embedding in self.embeddings.items()
            },
        }

        return output

    @classmethod
    def from_dict(cls, data: dict) -> "CodeUnit":
        """Create a CodeUnit from a dictionary."""
        if data["unit_type"] == "file":
            return File.from_dict(data)
        elif data["unit_type"] == "class":
            return Class.from_dict(data)
        elif data["unit_type"] == "function":
            return Function.from_dict(data)
        elif data["unit_type"] == "method":
            return Method.from_dict(data)
        else:
            raise ValueError(f"Invalid unit type: {data['unit_type']}")

    @abstractmethod
    def fully_qualified_name(self) -> str:
        """Get the fully qualified name of the code unit."""
        pass

    def __len__(self) -> int:
        """
        For code units which have sub-units (e.g. classes with methods),
        return the total number of sub-units, otherwise return 1.
        """
        return 1

    def __iter__(self) -> Iterator["CodeUnit"]:
        """
        For code units which have sub-units (e.g. classes with methods),
        iterate through all sub-units.
        """
        yield self

    def iter_flat(self) -> Iterator["CodeUnit"]:
        """
        Flat iteration yields self and then methods.
        This is useful for generating embeddings for all code units.
        """
        yield self


@dataclass
class TopLevelCodeUnit(CodeUnit):
    """
    Represents a top-level code unit (function or class).
    """

    file: "File" = None

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["filepath"] = str(self.file.filepath)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "TopLevelCodeUnit":
        """Create a TopLevelCodeUnit from a dictionary."""
        if data["unit_type"] == "class":
            return Class.from_dict(data)
        elif data["unit_type"] == "function":
            return Function.from_dict(data)

    def fully_qualified_name(self) -> str:
        """Get the fully qualified name of the code unit."""
        return f"{self.file.filepath}:{self.name}"


@dataclass
class File(CodeUnit, Iterable[TopLevelCodeUnit]):
    """Represents a file containing code."""

    unit_type = "file"
    filepath: Path = None
    codebase: str = None
    _code_units: List[TopLevelCodeUnit] = field(default_factory=list)

    @property
    def code_units(self) -> List[TopLevelCodeUnit]:
        """Get all code units in this file."""
        return self._code_units.copy()  # Return a copy to prevent direct modification

    @property
    def filename(self) -> str:
        """Get the filename from the filepath."""
        return self.filepath.name

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["filepath"] = str(self.filepath)
        result["codebase"] = self.codebase
        result["code_units"] = [unit.to_dict() for unit in self._code_units]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "File":
        """Create a File from a dictionary."""
        code_units_data = data.pop("code_units", [])
        file = cls(
            name=data["name"],
            source_code=data["source_code"],
            docstring=data.get("docstring"),
            id=data["id"],
            filepath=Path(data["filepath"]),
            codebase=data.get("codebase"),
            embeddings={
                model_name: CodeEmbedding.from_dict(embedding_data)
                for model_name, embedding_data in data["embeddings"].items()
            },
        )

        for unit_data in code_units_data:
            unit = TopLevelCodeUnit.from_dict(unit_data)
            file.add_code_unit(unit)

        return file

    def add_code_unit(self, unit: TopLevelCodeUnit) -> None:
        """Add a code unit to the file."""
        if unit.file is not None:
            unit.file.remove_code_unit(unit)  # Remove from old file if exists
        self._code_units.append(unit)
        unit.file = self  # Update the back reference

    def remove_code_unit(self, unit: TopLevelCodeUnit) -> None:
        """Remove a code unit from the file."""
        if unit in self._code_units:
            self._code_units.remove(unit)
            unit.file = None  # Clear the back reference

    def get_classes(self) -> List["Class"]:
        """Get all class definitions."""
        return [unit for unit in self._code_units if isinstance(unit, Class)]

    def get_functions(self) -> List["Function"]:
        """Get all standalone functions."""
        return [unit for unit in self._code_units if isinstance(unit, Function)]

    def __len__(self):
        return sum(len(unit) for unit in self._code_units)

    def __iter__(self) -> Iterator[TopLevelCodeUnit]:
        """
        Iterates through all code units in the file.
        This allows using: for unit in file
        """
        yield from self._code_units

    def iter_flat(self) -> Iterator[CodeUnit]:
        """Flat iteration yields all units and their methods."""
        for unit in self._code_units:
            yield from unit.iter_flat()

    def fully_qualified_name(self) -> str:
        """Get the fully qualified name of the code unit."""
        return str(self.filepath)


@dataclass
class Method(CodeUnit):
    """Represents a method within a class."""

    _class: "Class" = None
    unit_type = "method"

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["class"] = self._class.name
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Method":
        """Create a Method from a dictionary."""
        method = cls(
            name=data["name"],
            source_code=data["source_code"],
            docstring=data.get("docstring"),
            id=data["id"],
            embeddings={
                model_name: CodeEmbedding.from_dict(embedding_data)
                for model_name, embedding_data in data["embeddings"].items()
            },
        )
        return method

    @property
    def class_ref(self) -> "Class":
        return self._class

    @class_ref.setter
    def class_ref(self, value: "Class") -> None:
        self._class = value

    @property
    def filename(self) -> str:
        """Get the filename from the filepath."""
        return self.class_ref.file.filename

    @property
    def filepath(self) -> Path:
        """Get the filepath of the class."""
        return self.class_ref.file.filepath

    def fully_qualified_name(self) -> str:
        """Get the fully qualified name of the code unit."""
        return f"{self.class_ref.fully_qualified_name()}.{self.name}"


@dataclass
class Function(TopLevelCodeUnit):
    """Represents a standalone function."""

    unit_type = "function"

    @classmethod
    def from_dict(cls, data: dict) -> "Function":
        """Create a Function from a dictionary."""
        function = cls(
            name=data["name"],
            source_code=data["source_code"],
            docstring=data.get("docstring"),
            id=data["id"],
            embeddings={
                model_name: CodeEmbedding.from_dict(embedding_data)
                for model_name, embedding_data in data["embeddings"].items()
            },
        )
        return function


@dataclass
class Class(TopLevelCodeUnit):
    """Represents a class definition."""

    unit_type = "class"
    _methods: List[Method] = field(default_factory=list)

    @property
    def methods(self) -> List[Method]:
        """Get all methods in this class."""
        return self._methods.copy()  # Return a copy to prevent direct modification

    def add_method(self, method: Method) -> None:
        """Add a method to the class."""
        if method.class_ref is not None:
            # Remove from old class if exists
            method.class_ref.remove_method(method)
        self._methods.append(method)
        # Update the back reference
        method.class_ref = self

    def remove_method(self, method: Method) -> None:
        """Remove a method from the class."""
        if method in self._methods:
            self._methods.remove(method)
            # Clear the back reference
            method._class = None

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["methods"] = [method.to_dict() for method in self._methods]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Class":
        """Create a Class from a dictionary."""
        methods_data = data.pop("methods", [])
        new_class = cls(
            name=data["name"],
            source_code=data["source_code"],
            docstring=data.get("docstring"),
            id=data["id"],
            embeddings={
                model_name: CodeEmbedding.from_dict(embedding_data)
                for model_name, embedding_data in data["embeddings"].items()
            },
        )
        for method_data in methods_data:
            method = Method.from_dict(method_data)
            new_class.add_method(method)
        return new_class

    def __len__(self):
        return 1 + sum(len(method) for method in self._methods)

    def __iter__(self):
        """
        Iterates through all code units in the class, including methods.
        This allows using: for method in class
        """
        yield from self._methods

    def iter_flat(self) -> Iterator[CodeUnit]:
        """Flat iteration yields self and then methods."""
        yield self
        yield from self._methods
