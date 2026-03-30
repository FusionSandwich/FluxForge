"""Hardware abstraction layer foundations for future live acquisition support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class AcquisitionState(str, Enum):
    """Lifecycle state for an MCA-capable device."""

    DISCONNECTED = "disconnected"
    IDLE = "idle"
    ACQUIRING = "acquiring"
    FAULT = "fault"


@dataclass(frozen=True)
class DeviceStatus:
    """Snapshot of the current device state."""

    device_id: str
    label: str
    state: AcquisitionState
    message: str = ""
    live_time_s: float = 0.0
    real_time_s: float = 0.0
    high_voltage_v: float | None = None


class MCADevice(ABC):
    """Abstract base class for MCA devices."""

    def __init__(self, device_id: str, label: str) -> None:
        self.device_id = device_id
        self.label = label

    @abstractmethod
    def connect(self) -> DeviceStatus:
        """Open the device connection."""

    @abstractmethod
    def disconnect(self) -> DeviceStatus:
        """Close the device connection."""

    @abstractmethod
    def start_acquisition(self) -> DeviceStatus:
        """Begin acquisition."""

    @abstractmethod
    def stop_acquisition(self) -> DeviceStatus:
        """Stop acquisition."""

    @abstractmethod
    def read_counts(self) -> Sequence[float]:
        """Read the current spectrum counts."""

    @abstractmethod
    def status(self) -> DeviceStatus:
        """Return the current device status."""


class MockMCADevice(MCADevice):
    """Simple mock device used for early GUI and session scaffolding."""

    def __init__(self, device_id: str = "mock-mca", label: str = "Mock MCA") -> None:
        super().__init__(device_id=device_id, label=label)
        self._status = DeviceStatus(
            device_id=self.device_id,
            label=self.label,
            state=AcquisitionState.DISCONNECTED,
            message="Not connected",
        )
        self._counts = [0.0] * 4096

    def seed_counts(self, counts: Sequence[float]) -> None:
        """Replace the internal mock spectrum."""

        self._counts = [float(value) for value in counts]

    def connect(self) -> DeviceStatus:
        self._status = DeviceStatus(
            device_id=self.device_id,
            label=self.label,
            state=AcquisitionState.IDLE,
            message="Connected",
        )
        return self._status

    def disconnect(self) -> DeviceStatus:
        self._status = DeviceStatus(
            device_id=self.device_id,
            label=self.label,
            state=AcquisitionState.DISCONNECTED,
            message="Disconnected",
        )
        return self._status

    def start_acquisition(self) -> DeviceStatus:
        self._status = DeviceStatus(
            device_id=self.device_id,
            label=self.label,
            state=AcquisitionState.ACQUIRING,
            message="Acquiring",
        )
        return self._status

    def stop_acquisition(self) -> DeviceStatus:
        self._status = DeviceStatus(
            device_id=self.device_id,
            label=self.label,
            state=AcquisitionState.IDLE,
            message="Idle",
        )
        return self._status

    def read_counts(self) -> Sequence[float]:
        return tuple(self._counts)

    def status(self) -> DeviceStatus:
        return self._status
