"""Hardware abstraction layer foundations for future live acquisition support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


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

    def spectrum_source_fields(self) -> dict[str, str]:
        """Return standardized spectrum-source metadata for session models."""

        return {
            "source_type": "hal",
            "device_id": self.device_id,
            "device_label": self.label,
        }


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


class DeviceRegistry:
    """Registry for discoverable MCA-capable devices."""

    def __init__(self, devices: Iterable[MCADevice] = ()) -> None:
        self._devices: dict[str, MCADevice] = {}
        self._default_device_id: str | None = None
        for device in devices:
            self.register(device)

    def register(self, device: MCADevice, *, default: bool = False) -> MCADevice:
        """Register a device and optionally make it the default selection."""

        self._devices[device.device_id] = device
        if default or self._default_device_id is None:
            self._default_device_id = device.device_id
        return device

    def unregister(self, device_id: str) -> MCADevice | None:
        """Remove a device from the registry."""

        removed = self._devices.pop(device_id, None)
        if self._default_device_id == device_id:
            self._default_device_id = next(iter(self._devices), None)
        return removed

    def get(self, device_id: str) -> MCADevice | None:
        """Fetch a device by id."""

        return self._devices.get(device_id)

    def default(self) -> MCADevice | None:
        """Return the default device, when one is available."""

        if self._default_device_id is None:
            return None
        return self._devices.get(self._default_device_id)

    def devices(self) -> tuple[MCADevice, ...]:
        """Return all registered devices in insertion order."""

        return tuple(self._devices.values())

    def keys(self) -> tuple[str, ...]:
        """Return registered device ids."""

        return tuple(self._devices.keys())

    def snapshot(self) -> list[dict[str, str]]:
        """Return a serializable snapshot of the registry."""

        records = []
        for device in self._devices.values():
            record = device.spectrum_source_fields()
            record["state"] = device.status().state.value
            records.append(record)
        return records
