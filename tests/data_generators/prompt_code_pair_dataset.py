import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from uuid import uuid4

from storage.code_store import CodeUnit, CodebaseSnapshot


@dataclass
class PromptCodePair:
    """
    Represents a single prompt-code-unit pair in the dataset.
    Contains both the prompt and the reference to the original code unit.
    """

    prompt: str
    code_unit: CodeUnit
    id: str = field(default_factory=lambda: str(uuid4()))
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @property
    def unit_type(self) -> str:
        """Get the type of the code unit"""
        return self.code_unit.unit_type

    def to_dict(self) -> dict:
        """Convert the prompt-code pair to a dictionary for serialization."""
        return {
            "id": self.id,
            "generated_at": self.generated_at,
            "prompt": self.prompt,
            "code": self.code_unit.source_code,
            "unit_type": self.unit_type,
            "code_unit_id": self.code_unit.id,
            "code_unit_name": self.code_unit.name,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        code_unit: CodeUnit,
    ) -> "PromptCodePair":
        """Create a PromptCodePair from a dictionary."""
        return cls(
            prompt=data["prompt"],
            code_unit=code_unit,
            id=data.get("id", str(uuid4())),
            generated_at=data.get("generated_at", datetime.now().isoformat())
        )

@dataclass
class PromptCodePairDataset:
    """
    Collection of prompt-code pairs with methods for management.
    """

    _pairs: List[PromptCodePair] = field(default_factory=list)
    _id_index: Dict[str, PromptCodePair] = field(default_factory=dict,
                                                 init=False)
    _code_unit_index: Dict[str, PromptCodePair] = field(default_factory=dict,
                                                        init=False)

    def __post_init__(self):
        """Initialize indices for efficient lookups."""
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild the ID and code unit indices."""
        self._id_index.clear()
        self._code_unit_index.clear()

        for pair in self._pairs:
            self._id_index[pair.id] = pair
            self._code_unit_index[pair.code_unit.id] = pair

    def add_pair(self, pair: PromptCodePair) -> None:
        """Add a prompt-code pair to the dataset."""
        self._pairs.append(pair)
        self._id_index[pair.id] = pair
        self._code_unit_index[pair.code_unit.id] = pair

    def remove_pair(self, pair_id: str) -> Optional[PromptCodePair]:
        """Remove a prompt-code pair from the dataset by its ID."""
        pair = self._id_index.get(pair_id)
        if pair:
            self._pairs.remove(pair)
            del self._id_index[pair_id]
            del self._code_unit_index[pair.code_unit.id]
            return pair
        return None

    def get_pair_by_id(self, pair_id: str) -> Optional[PromptCodePair]:
        """Retrieve a prompt-code pair by its ID."""
        return self._id_index.get(pair_id)

    def get_pair_by_code_unit(self, code_unit_id: str) -> Optional[PromptCodePair]:
        """Retrieve a prompt-code pair by its code unit ID."""
        return self._code_unit_index.get(code_unit_id)

    def get_pairs_by_type(self, unit_type: str) -> List[PromptCodePair]:
        """Get all pairs of a specific unit type."""
        return [pair for pair in self._pairs if pair.unit_type == unit_type]

    def to_json(self, json_path: Path) -> None:
        """Save the dataset to a JSON file."""
        data = [pair.to_dict() for pair in self._pairs]
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(
        cls,
        json_path: Path,
        codebase: CodebaseSnapshot
    ) -> "PromptCodePairDataset":
        """
        Load a dataset from a JSON file.

        Args:
            json_path: Path to the JSON file
            codebase: CodebaseSnapshot to link code units
        """
        dataset = cls()
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for pair_data in data:
            code_unit = codebase.get_unit_by_id(pair_data["code_unit_id"])
            pair = PromptCodePair.from_dict(pair_data, code_unit)
            dataset.add_pair(pair)

        return dataset

    def filter(self, predicate) -> "PromptCodePairDataset":
        """Create a new dataset with pairs that match the predicate."""
        new_dataset = PromptCodePairDataset()
        for pair in self._pairs:
            if predicate(pair):
                new_dataset.add_pair(pair)
        return new_dataset

    def __len__(self) -> int:
        """Get the number of pairs in the dataset."""
        return len(self._pairs)

    def __iter__(self) -> Iterator[PromptCodePair]:
        """Iterate through all pairs in the dataset."""
        yield from self._pairs
