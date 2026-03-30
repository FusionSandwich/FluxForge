"""State manager for selectable data libraries in the modern GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fluxforge.data.nuclear_data_sources import (
    NuclearDataSourceRecord,
    get_nuclear_data_source,
    list_nuclear_data_sources,
    summarize_nuclear_data_source,
)


@dataclass(frozen=True)
class DataLibraryState:
    """Serializable selection state for GUI-accessible data libraries."""

    gamma_identification_source_id: str = "fluxforge_bundled_gamma"
    calibration_source_id: str = "calibration_standard_sources"
    naa_monitor_source_id: str = "k0_naa_monitors"
    dosimetry_source_id: str = "irdff_ii_dosimetry"
    activation_catalog_source_id: str = "flux_wire_catalog"
    custom_gamma_path: str | None = None


LibraryListener = Callable[[DataLibraryState], None]


class DataLibraryManager:
    """Persist and publish active data-library selections for the Qt shell."""

    GAMMA_SOURCE_KEY = "gui/data_libraries/gamma_identification_source"
    CALIBRATION_SOURCE_KEY = "gui/data_libraries/calibration_source"
    NAA_MONITOR_SOURCE_KEY = "gui/data_libraries/naa_monitor_source"
    DOSIMETRY_SOURCE_KEY = "gui/data_libraries/dosimetry_source"
    ACTIVATION_SOURCE_KEY = "gui/data_libraries/activation_source"
    CUSTOM_GAMMA_PATH_KEY = "gui/data_libraries/custom_gamma_path"

    def __init__(
        self,
        *,
        initial_state: DataLibraryState | None = None,
        settings=None,
    ) -> None:
        self._settings = settings
        self._state = initial_state or self._load_state()
        self._listeners: list[LibraryListener] = []

    @property
    def state(self) -> DataLibraryState:
        return self._state

    def subscribe(self, listener: LibraryListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: LibraryListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _load_state(self) -> DataLibraryState:
        if self._settings is None:
            return DataLibraryState()
        return DataLibraryState(
            gamma_identification_source_id=str(
                self._settings_value(
                    self.GAMMA_SOURCE_KEY,
                    DataLibraryState.gamma_identification_source_id,
                )
            ),
            calibration_source_id=str(
                self._settings_value(
                    self.CALIBRATION_SOURCE_KEY,
                    DataLibraryState.calibration_source_id,
                )
            ),
            naa_monitor_source_id=str(
                self._settings_value(
                    self.NAA_MONITOR_SOURCE_KEY,
                    DataLibraryState.naa_monitor_source_id,
                )
            ),
            dosimetry_source_id=str(
                self._settings_value(
                    self.DOSIMETRY_SOURCE_KEY,
                    DataLibraryState.dosimetry_source_id,
                )
            ),
            activation_catalog_source_id=str(
                self._settings_value(
                    self.ACTIVATION_SOURCE_KEY,
                    DataLibraryState.activation_catalog_source_id,
                )
            ),
            custom_gamma_path=self._normalize_optional_text(
                self._settings_value(self.CUSTOM_GAMMA_PATH_KEY, None)
            ),
        )

    def _save_state(self, state: DataLibraryState) -> None:
        if self._settings is None:
            return
        self._settings_set_value(
            self.GAMMA_SOURCE_KEY, state.gamma_identification_source_id
        )
        self._settings_set_value(
            self.CALIBRATION_SOURCE_KEY, state.calibration_source_id
        )
        self._settings_set_value(self.NAA_MONITOR_SOURCE_KEY, state.naa_monitor_source_id)
        self._settings_set_value(self.DOSIMETRY_SOURCE_KEY, state.dosimetry_source_id)
        self._settings_set_value(
            self.ACTIVATION_SOURCE_KEY, state.activation_catalog_source_id
        )
        self._settings_set_value(self.CUSTOM_GAMMA_PATH_KEY, state.custom_gamma_path or "")
        sync = getattr(self._settings, "sync", None)
        if callable(sync):
            sync()

    def _publish(self, state: DataLibraryState) -> DataLibraryState:
        self._state = state
        self._save_state(state)
        for listener in tuple(self._listeners):
            listener(state)
        return state

    def _settings_value(self, key: str, default):
        getter = getattr(self._settings, "value", None)
        if callable(getter):
            return getter(key, default)
        return default

    def _settings_set_value(self, key: str, value) -> None:
        setter = getattr(self._settings, "setValue", None)
        if callable(setter):
            setter(key, value)

    def _normalize_optional_text(self, value) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def set_gamma_identification_source(
        self,
        source_id: str,
        *,
        custom_gamma_path: str | None = None,
    ) -> DataLibraryState:
        return self._publish(
            DataLibraryState(
                gamma_identification_source_id=source_id,
                calibration_source_id=self._state.calibration_source_id,
                naa_monitor_source_id=self._state.naa_monitor_source_id,
                dosimetry_source_id=self._state.dosimetry_source_id,
                activation_catalog_source_id=self._state.activation_catalog_source_id,
                custom_gamma_path=self._normalize_optional_text(
                    custom_gamma_path
                    if custom_gamma_path is not None
                    else self._state.custom_gamma_path
                ),
            )
        )

    def set_custom_gamma_path(self, custom_gamma_path: str | None) -> DataLibraryState:
        return self._publish(
            DataLibraryState(
                gamma_identification_source_id=self._state.gamma_identification_source_id,
                calibration_source_id=self._state.calibration_source_id,
                naa_monitor_source_id=self._state.naa_monitor_source_id,
                dosimetry_source_id=self._state.dosimetry_source_id,
                activation_catalog_source_id=self._state.activation_catalog_source_id,
                custom_gamma_path=self._normalize_optional_text(custom_gamma_path),
            )
        )

    def set_calibration_source(self, source_id: str) -> DataLibraryState:
        return self._publish(
            DataLibraryState(
                gamma_identification_source_id=self._state.gamma_identification_source_id,
                calibration_source_id=source_id,
                naa_monitor_source_id=self._state.naa_monitor_source_id,
                dosimetry_source_id=self._state.dosimetry_source_id,
                activation_catalog_source_id=self._state.activation_catalog_source_id,
                custom_gamma_path=self._state.custom_gamma_path,
            )
        )

    def set_naa_monitor_source(self, source_id: str) -> DataLibraryState:
        return self._publish(
            DataLibraryState(
                gamma_identification_source_id=self._state.gamma_identification_source_id,
                calibration_source_id=self._state.calibration_source_id,
                naa_monitor_source_id=source_id,
                dosimetry_source_id=self._state.dosimetry_source_id,
                activation_catalog_source_id=self._state.activation_catalog_source_id,
                custom_gamma_path=self._state.custom_gamma_path,
            )
        )

    def set_dosimetry_source(self, source_id: str) -> DataLibraryState:
        return self._publish(
            DataLibraryState(
                gamma_identification_source_id=self._state.gamma_identification_source_id,
                calibration_source_id=self._state.calibration_source_id,
                naa_monitor_source_id=self._state.naa_monitor_source_id,
                dosimetry_source_id=source_id,
                activation_catalog_source_id=self._state.activation_catalog_source_id,
                custom_gamma_path=self._state.custom_gamma_path,
            )
        )

    def set_activation_catalog_source(self, source_id: str) -> DataLibraryState:
        return self._publish(
            DataLibraryState(
                gamma_identification_source_id=self._state.gamma_identification_source_id,
                calibration_source_id=self._state.calibration_source_id,
                naa_monitor_source_id=self._state.naa_monitor_source_id,
                dosimetry_source_id=self._state.dosimetry_source_id,
                activation_catalog_source_id=source_id,
                custom_gamma_path=self._state.custom_gamma_path,
            )
        )

    def available_sources(self, category: str) -> tuple[NuclearDataSourceRecord, ...]:
        records = list_nuclear_data_sources(self._custom_paths())
        if category == "gamma_identification":
            return tuple(
                record
                for record in records
                if "peak-identification" in record.capabilities
                or record.source_id.startswith("custom_gamma_")
                or record.source_id == "custom_gamma_file"
            )
        if category == "calibration":
            return tuple(
                record for record in records if "efficiency-calibration" in record.capabilities
            )
        if category == "naa_monitor":
            return tuple(record for record in records if "k0-naa" in record.capabilities)
        if category == "dosimetry":
            return tuple(record for record in records if "dosimetry" in record.capabilities)
        if category == "activation":
            return tuple(
                record
                for record in records
                if "activation-reference" in record.capabilities
                and record.source_id != "nndc_offline_activation"
            )
        raise KeyError(f"Unknown data-library category: {category}")

    def record_for_category(self, category: str) -> NuclearDataSourceRecord:
        state = self._state
        selected_id = {
            "gamma_identification": state.gamma_identification_source_id,
            "calibration": state.calibration_source_id,
            "naa_monitor": state.naa_monitor_source_id,
            "dosimetry": state.dosimetry_source_id,
            "activation": state.activation_catalog_source_id,
        }[category]
        return get_nuclear_data_source(selected_id, custom_paths=self._custom_paths())

    def summary_for_category(self, category: str) -> str:
        record = self.record_for_category(category)
        custom_path = (
            self._state.custom_gamma_path if record.source_id == "custom_gamma_file" else None
        )
        return summarize_nuclear_data_source(record.source_id, custom_path=custom_path)

    def describe(self) -> dict[str, str | None]:
        return {
            "gamma_identification_source_id": self._state.gamma_identification_source_id,
            "calibration_source_id": self._state.calibration_source_id,
            "naa_monitor_source_id": self._state.naa_monitor_source_id,
            "dosimetry_source_id": self._state.dosimetry_source_id,
            "activation_catalog_source_id": self._state.activation_catalog_source_id,
            "custom_gamma_path": self._state.custom_gamma_path,
        }

    def _custom_paths(self) -> tuple[str, ...]:
        return (self._state.custom_gamma_path,) if self._state.custom_gamma_path else ()


__all__ = ["DataLibraryManager", "DataLibraryState"]
