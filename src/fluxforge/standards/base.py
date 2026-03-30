"""Standards-module scaffolding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LockedSetting:
    """Represents a standards-locked parameter."""

    field_id: str
    value: str
    standard_section: str


class StandardsModule(ABC):
    """Base class for standards-constrained workflows."""

    standard_id: str
    display_name: str

    @abstractmethod
    def locked_settings(self) -> tuple[LockedSetting, ...]:
        """Return the parameters locked by this standard."""
