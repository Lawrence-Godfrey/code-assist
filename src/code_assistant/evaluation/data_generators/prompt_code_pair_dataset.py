import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from uuid import uuid4

from code_assistant.storage.code_store import CodebaseSnapshot, CodeUnit


@dataclass
class SplitConfig:
    """Configuration for dataset splitting"""

    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: Optional[int] = None  # Specify random seed for reproducibility

    def __post_init__(self):
        """
        Validate split ratios.

        Ensures:
        1. All ratios sum to 1.0
        2. Train ratio is greater than 0
        3. Test ratio is greater than 0
        4. Validation ratio is non-negative
        """
        # Check ratios sum to 1
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-10:  # Using epsilon for float comparison
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        # Check train ratio
        if self.train_ratio <= 0:
            raise ValueError(
                f"Train ratio must be greater than 0, got {self.train_ratio}"
            )

        # Check test ratio
        if self.test_ratio <= 0:
            raise ValueError(
                f"Test ratio must be greater than 0, got {self.test_ratio}"
            )

        # Check validation ratio is non-negative
        if self.validation_ratio < 0:
            raise ValueError(
                f"Validation ratio cannot be negative, got {self.validation_ratio}"
            )

    def save_config(
        self,
        output_dir: Path,
        train_samples: int,
        validation_samples: int,
        test_samples: int,
    ) -> None:
        """
        Save split configuration and results to JSON file.

        Args:
            output_dir: Directory to save configuration
            train_samples: Number of training samples
            validation_samples: Number of validation samples
            test_samples: Number of test samples
        """
        total_samples = train_samples + validation_samples + test_samples

        config_data = {
            "ratios": {
                "train_ratio": self.train_ratio,
                "validation_ratio": self.validation_ratio,
                "test_ratio": self.test_ratio,
            },
            "random_seed": self.random_seed,
            "samples": {
                "total_samples": total_samples,
                "train_samples": train_samples,
                "validation_samples": validation_samples,
                "test_samples": test_samples,
            },
            "created_at": datetime.now().isoformat(),
        }

        config_path = output_dir / "split_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)


@dataclass
class PromptCodePair:
    """
    Represents a single prompt-code-unit pair in the dataset.
    Contains both the prompt and the reference to the original code unit.
    """

    prompt: str
    code_unit: CodeUnit
    id: str = field(default_factory=lambda: str(uuid4()))
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

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
            generated_at=data.get("generated_at", datetime.now().isoformat()),
        )


@dataclass
class PromptCodePairDataset:
    """
    Collection of prompt-code pairs with methods for management.
    """

    _pairs: List[PromptCodePair] = field(default_factory=list)
    _id_index: Dict[str, PromptCodePair] = field(default_factory=dict, init=False)
    _code_unit_index: Dict[str, PromptCodePair] = field(
        default_factory=dict, init=False
    )

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

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(
        cls, json_path: Path, codebase: CodebaseSnapshot
    ) -> "PromptCodePairDataset":
        """
        Load a dataset from a JSON file.

        Args:
            json_path: Path to the JSON file
            codebase: CodebaseSnapshot to link code units
        """
        dataset = cls()
        with open(json_path, "r", encoding="utf-8") as f:
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

    def create_splits(
        self, output_dir: Path, config: Optional[SplitConfig] = None
    ) -> Dict[str, "PromptCodePairDataset"]:
        """
        Create dataset splits based on provided configuration and save them.

        Args:
            output_dir: Directory where split datasets will be saved
            config: Configuration for splitting. If None, uses default ratios

        Returns:
            Dictionary containing the created datasets with keys:
            'train', 'validation' (if any), and 'test'
        """
        if config is None:
            config = SplitConfig()

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed if provided
        if config.random_seed is not None:
            random.seed(config.random_seed)

        # Shuffle pairs
        all_pairs = self._pairs.copy()
        random.shuffle(all_pairs)

        # Calculate split indices
        total_size = len(all_pairs)
        train_size = int(total_size * config.train_ratio)
        val_size = int(total_size * config.validation_ratio)
        test_size = total_size - train_size - val_size

        # Initialize result dictionary
        split_datasets = {}

        # Create and save train dataset
        train_pairs = all_pairs[:train_size]
        train_dataset = PromptCodePairDataset()
        for pair in train_pairs:
            train_dataset.add_pair(pair)
        train_dataset.to_json(output_dir / "train_prompt_code_pair_dataset.json")
        split_datasets["train"] = train_dataset

        # Create validation dataset if validation ratio > 0
        if config.validation_ratio > 0:
            val_pairs = all_pairs[train_size : train_size + val_size]
            val_dataset = PromptCodePairDataset()
            for pair in val_pairs:
                val_dataset.add_pair(pair)
            val_dataset.to_json(output_dir / "validate_prompt_code_pair_dataset.json")
            split_datasets["validation"] = val_dataset

        test_pairs = all_pairs[train_size + val_size :]
        test_dataset = PromptCodePairDataset()
        for pair in test_pairs:
            test_dataset.add_pair(pair)
        test_dataset.to_json(output_dir / "test_prompt_code_pair_dataset.json")
        split_datasets["test"] = test_dataset

        # Save configuration
        config.save_config(
            output_dir,
            train_samples=train_size,
            validation_samples=val_size,
            test_samples=test_size,
        )

        return split_datasets

    @staticmethod
    def load_splits(
        split_dir: Path, codebase: CodebaseSnapshot
    ) -> Dict[str, "PromptCodePairDataset"]:
        """
        Load previously created dataset splits.

        Args:
            split_dir: Directory containing the split datasets
            codebase: Codebase to link code units

        Returns:
            Dictionary containing the available datasets with keys: 'train', 'validation' (if present), 'test'
        """
        split_dir = Path(split_dir)
        splits = {}

        # Load training dataset (required)
        train_path = split_dir / "train_prompt_code_pair_dataset.json"
        if not train_path.exists():
            raise FileNotFoundError("Training dataset not found in splits directory")

        # Load testing dataset (required)
        test_path = split_dir / "test_prompt_code_pair_dataset.json"
        if not test_path.exists():
            raise FileNotFoundError("Test dataset not found in splits directory")

        splits["train"] = PromptCodePairDataset.from_json(train_path, codebase)
        splits["test"] = PromptCodePairDataset.from_json(test_path, codebase)

        # Try loading validation dataset
        val_path = split_dir / "validate_prompt_code_pair_dataset.json"
        if val_path.exists():
            splits["validation"] = PromptCodePairDataset.from_json(val_path, codebase)

        return splits

    def __len__(self) -> int:
        """Get the number of pairs in the dataset."""
        return len(self._pairs)

    def __iter__(self) -> Iterator[PromptCodePair]:
        """Iterate through all pairs in the dataset."""
        yield from self._pairs
