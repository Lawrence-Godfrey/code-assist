from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import json
from abc import ABC


@dataclass
class CodeUnit(ABC):
    """
    Abstract base class for all code units (classes, methods, functions).
    Cannot be instantiated directly - must use a concrete subclass.
    """

    name: str
    source_code: str
    docstring: Optional[str] = None
    unit_type: str = None

    def to_dict(self) -> dict:
        """Convert the code unit to a dictionary."""
        return {
            "type": self.unit_type,
            "name": self.name,
            "source_code": self.source_code,
            "docstring": self.docstring,
        }

    def __len__(self) -> int:
        """
        For code units which have sub-units (e.g. classes with methods),
        return the total number of sub-units, otherwise return 1.
        """
        return 1

    def __iter__(self):
        """
        For code units which have sub-units (e.g. classes with methods),
        iterate through all sub-units.
        """
        yield self


@dataclass
class File(CodeUnit):
    """Represents a file containing code."""

    unit_type = "file"
    filepath: Path = None
    _code_units: List[CodeUnit] = field(default_factory=list)  # Make private

    @property
    def code_units(self) -> List[CodeUnit]:
        """Get all code units in this file."""
        return self._code_units.copy()  # Return a copy to prevent direct modification

    @property
    def filename(self) -> str:
        """Get the filename from the filepath."""
        return self.filepath.name

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["filepath"] = str(self.filepath)
        result["code_units"] = [unit.to_dict() for unit in self._code_units]
        return result

    def add_code_unit(self, unit: CodeUnit) -> None:
        """Add a code unit to the file."""
        if unit.file is not None:
            unit.file.remove_code_unit(unit)  # Remove from old file if exists
        self._code_units.append(unit)
        unit.file = self  # Update the back reference

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
        Iterates through all code units in the file.
        This allows using: for unit in file
        """
        for unit in self._code_units:
            yield from unit


@dataclass
class Method(CodeUnit):
    """Represents a method within a class."""

    _class: "Class" = None
    unit_type = "method"

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["class"] = self._class
        return result

    @property
    def class_ref(self) -> "Class":
        return self._class

    @class_ref.setter
    def class_ref(self, value: "Class") -> None:
        self._class = value


@dataclass
class Function(CodeUnit):
    """Represents a standalone function."""

    unit_type = "function"
    file: File = None


@dataclass
class Class(CodeUnit):
    """Represents a class definition."""

    unit_type = "class"
    _methods: List[Method] = field(default_factory=list)  # Make private
    file: File = None

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

    def __len__(self):
        return 1 + sum(len(method) for method in self._methods)

    def __iter__(self):
        """
        Iterates through all code units in the class, including methods.
        This allows using: for unit in class
        """
        yield self
        yield from self._methods


@dataclass
class CodebaseSnapshot:
    """Collection of code units representing a codebase at a point in time."""

    _files: List[File] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_path: Path) -> "CodebaseSnapshot":
        """Load code units from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: List[dict]) -> "CodebaseSnapshot":
        """Create a CodebaseSnapshot from a list of dictionaries."""
        files = []
        for file_data in data:
            file_data["filepath"] = Path(file_data["filepath"])
            file = File(**file_data)
            code_units_data = file_data.pop("code_units", [])
            for unit_data in code_units_data:
                unit_type = unit_data["type"]
                if unit_type == "class":
                    file.add_code_unit(Class(**unit_data))
                elif unit_type == "function":
                    file.add_code_unit(Function(**unit_data))

            files.append(file)

        return cls(files)

    def to_json(self, json_path: Path) -> None:
        """Save code units to a JSON file."""
        data = [file.to_dict() for file in self.files]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @property
    def files(self) -> List[File]:
        """Get all files in the codebase."""
        return self._files.copy()  # Return a copy to prevent direct modification

    def add_file(self, file: File) -> None:
        """Add a file to the codebase snapshot."""
        self._files.append(file)

    def __len__(self) -> int:
        """
        Returns the total number of code units across all files.
        This allows using len(codebase_snapshot).
        """
        return sum(len(file.code_units) for file in self._files)

    def __iter__(self):
        """
        Iterates through all code units in the codebase, including methods within classes.
        This allows using: for unit in codebase_snapshot
        """
        yield from self._files
