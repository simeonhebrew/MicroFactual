"""Core data structures and base classes for microfactual."""

from .base import BaseModel
from .dataset import MicrobiomeDataset

__all__ = ["MicrobiomeDataset", "BaseModel"]
