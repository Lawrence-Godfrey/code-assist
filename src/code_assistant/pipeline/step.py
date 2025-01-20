"""
This module defines the base class for all pipeline steps.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    def __init__(self) -> None:
        self._next_step: Optional[PipelineStep] = None

    def set_next(self, step: 'PipelineStep') -> 'PipelineStep':
        """Set the next step in the pipeline."""
        self._next_step = step
        return step

    def execute_next(self, context: Dict) -> None:
        """Execute the next step in the pipeline if it exists."""
        if self._next_step:
            return self._next_step.execute(context)
        return None

    @abstractmethod
    def execute(self, context: Dict) -> None:
        """Execute this pipeline step."""
        pass