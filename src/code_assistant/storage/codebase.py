import uuid
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from code_assistant.logging.logger import get_logger
from code_assistant.storage.stores.base import EmbeddingUnit

logger = get_logger(__name__)


@dataclass
class CodeUnit(ABC):
    """
    Abstract base class for all code units (classes, methods, functions).
    Cannot be instantiated directly - must use a concrete subclass.
    """

    name: str
    source_code: str
    codebase: str
    filepath: str
    unit_type: str = field(init=False)
    docstring: Optional[str] = None
    embeddings: Dict[str, EmbeddingUnit] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        """Convert the code unit to a dictionary."""
        output = {
            "id": self.id,
            "unit_type": self.unit_type,
            "name": self.name,
            "codebase": self.codebase,
            "filepath": str(self.filepath),
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
        unit_type = data.pop("unit_type")
        if unit_type == "file":
            return File.from_dict(data)
        elif unit_type == "class":
            return Class.from_dict(data)
        elif unit_type == "function":
            return Function.from_dict(data)
        elif unit_type == "method":
            return Method.from_dict(data)
        else:
            raise ValueError(f"Invalid unit type: {data['unit_type']}")

    def fully_qualified_name(self) -> str:
        """Get the fully qualified name of the code unit."""
        return f"{self.filepath}:{self.name}"

    @property
    def filename(self) -> str:
        """Get the filename from the filepath."""
        return Path(self.filepath).name

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
class File(CodeUnit, Iterable[CodeUnit]):
    """Represents a file containing code."""

    unit_type = "file"
    filepath: Path = None
    _code_units: List[CodeUnit] = field(default_factory=list)

    @property
    def code_units(self) -> List[CodeUnit]:
        """Get all code units in this file."""
        return self._code_units.copy()  # Return a copy to prevent direct modification

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["code_units"] = [unit.to_dict() for unit in self._code_units]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "File":
        """Create a File from a dictionary."""
        code_units_data = data.pop("code_units", [])
        file = cls(**data)

        for unit_data in code_units_data:
            file.add_code_unit(**unit_data)

        return file

    def add_code_unit(self, unit: CodeUnit) -> None:
        """Add a code unit to the file."""
        self._code_units.append(unit)

    def remove_code_unit(self, unit: CodeUnit) -> None:
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

    def __iter__(self):
        """
        Iterates through all code units in the file, including classes and functions.
        This allows using: for unit in file
        """
        yield from self._code_units

    def iter_flat(self) -> Iterator[CodeUnit]:
        """Flat iteration yields all units and their methods."""
        for unit in self._code_units:
            yield from unit.iter_flat()


@dataclass
class Method(CodeUnit):
    """Represents a method within a class."""

    unit_type = "method"
    classname: str = None

    def __post_init__(self):
        if self.classname is None:
            raise ValueError("Method must have a classname")

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["classname"] = self.classname
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Method":
        """Create a Method from a dictionary."""
        method = cls(**data)
        return method

    def fully_qualified_name(self) -> str:
        """Get the fully qualified name of the code unit."""
        return f"{self.filepath}:{self.classname}.{self.name}"


@dataclass
class Function(CodeUnit):
    """Represents a standalone function."""

    unit_type = "function"

    @classmethod
    def from_dict(cls, data: dict) -> "CodeUnit":
        """Create a CodeUnit from a dictionary."""
        code_unit = cls(
            **data,
        )
        return code_unit


@dataclass
class Class(CodeUnit):
    """Represents a class definition."""

    unit_type = "class"
    _methods: List[Method] = field(default_factory=list)

    @property
    def methods(self) -> List[Method]:
        """Get all methods in this class."""
        return self._methods.copy()  # Return a copy to prevent direct modification

    def add_method(self, method: Method) -> None:
        """Add a method to the class."""
        self._methods.append(method)

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
            **data,
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
