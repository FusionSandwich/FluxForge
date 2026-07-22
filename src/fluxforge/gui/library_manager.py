"""State manager for selectable data libraries in the modern GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fluxforge.data.nuclear_data_sources import (
    NuclearDataSourceRecord,
    get_nuclear_data_source,
    list_nuclear_data_sources,
    list_registered_user_gamma_sources,
    register_user_gamma_source,
    remove_user_gamma_source,
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


STANDARD_LIBRARY_LOCKS: dict[str, dict[str, str]] = {
    "ASTM E181": {
        "gamma_identification": "decay_2012",
        "calibration": "calibration_standard_sources",
    },
    "ASTM E1218": {
        "gamma_identification": "decay_2012",
        "calibration": "calibration_standard_sources",
    },
    "ASTM E1297": {
        "gamma_identification": "decay_2012",
        "calibration": "calibration_standard_sources",
    },
    "ASTM C1232": {
        "gamma_identification": "decay_2012",
        "calibration": "calibration_standard_sources",
    },
    "ASTM C1030": {
        "gamma_identification": "decay_2012",
        "calibration": "calibration_standard_sources",
    },
    "ASTM E261": {
        "gamma_identification": "decay_2012",
        "calibration": "calibration_standard_sources",
        "dosimetry": "irdff_ii_dosimetry",
        "activation": "flux_wire_catalog",
    },
}


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
        self._state = self._validated_state(initial_state or self._load_state())
        self._listeners: list[LibraryListener] = []
        if self._settings is not None:
            self._save_state(self._state)

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

    def _validated_state(self, state: DataLibraryState) -> DataLibraryState:
        """Recover from removed or unavailable user gamma-library selections."""

        custom_paths = [state.custom_gamma_path] if state.custom_gamma_path else ()
        try:
            get_nuclear_data_source(
                state.gamma_identification_source_id,
                custom_paths=custom_paths,
            )
        except (KeyError, OSError, ValueError):
            return DataLibraryState(
                gamma_identification_source_id="fluxforge_bundled_gamma",
                calibration_source_id=state.calibration_source_id,
                naa_monitor_source_id=state.naa_monitor_source_id,
                dosimetry_source_id=state.dosimetry_source_id,
                activation_catalog_source_id=state.activation_catalog_source_id,
                custom_gamma_path=None,
            )
        return state

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

    def locked_source_for_category(
        self,
        category: str,
        *,
        standard: str | None = None,
    ) -> str | None:
        if not standard:
            return None
        return STANDARD_LIBRARY_LOCKS.get(str(standard), {}).get(category)

    def resolved_state(
        self,
        *,
        standard: str | None = None,
    ) -> DataLibraryState:
        state = self._state
        return DataLibraryState(
            gamma_identification_source_id=(
                self.locked_source_for_category(
                    "gamma_identification",
                    standard=standard,
                )
                or state.gamma_identification_source_id
            ),
            calibration_source_id=(
                self.locked_source_for_category("calibration", standard=standard)
                or state.calibration_source_id
            ),
            naa_monitor_source_id=(
                self.locked_source_for_category("naa_monitor", standard=standard)
                or state.naa_monitor_source_id
            ),
            dosimetry_source_id=(
                self.locked_source_for_category("dosimetry", standard=standard)
                or state.dosimetry_source_id
            ),
            activation_catalog_source_id=(
                self.locked_source_for_category("activation", standard=standard)
                or state.activation_catalog_source_id
            ),
            custom_gamma_path=state.custom_gamma_path,
        )

    def available_sources(
        self,
        category: str,
        *,
        standard: str | None = None,
    ) -> tuple[NuclearDataSourceRecord, ...]:
        records = list_nuclear_data_sources(self._custom_paths())
        locked_source_id = self.locked_source_for_category(category, standard=standard)
        if category == "gamma_identification":
            resolved = tuple(
                record
                for record in records
                if "peak-identification" in record.capabilities
                or record.source_id.startswith("custom_gamma_")
                or record.source_id == "custom_gamma_file"
            )
            if locked_source_id:
                return tuple(
                    record for record in resolved if record.source_id == locked_source_id
                )
            return resolved
        if category == "calibration":
            resolved = tuple(
                record for record in records if "efficiency-calibration" in record.capabilities
            )
            if locked_source_id:
                return tuple(
                    record for record in resolved if record.source_id == locked_source_id
                )
            return resolved
        if category == "naa_monitor":
            resolved = tuple(record for record in records if "k0-naa" in record.capabilities)
            if locked_source_id:
                return tuple(
                    record for record in resolved if record.source_id == locked_source_id
                )
            return resolved
        if category == "dosimetry":
            resolved = tuple(
                record for record in records if "dosimetry" in record.capabilities
            )
            if locked_source_id:
                return tuple(
                    record for record in resolved if record.source_id == locked_source_id
                )
            return resolved
        if category == "activation":
            resolved = tuple(
                record
                for record in records
                if "activation-reference" in record.capabilities
                and record.source_id != "nndc_offline_activation"
            )
            resolved = tuple(
                sorted(
                    resolved,
                    key=lambda record: (
                        0 if record.source_id == "flux_wire_catalog" else 1,
                        record.label.lower(),
                    ),
                )
            )
            if locked_source_id:
                return tuple(
                    record for record in resolved if record.source_id == locked_source_id
                )
            return resolved
        raise KeyError(f"Unknown data-library category: {category}")

    def record_for_category(
        self,
        category: str,
        *,
        standard: str | None = None,
    ) -> NuclearDataSourceRecord:
        state = self.resolved_state(standard=standard)
        selected_id = {
            "gamma_identification": state.gamma_identification_source_id,
            "calibration": state.calibration_source_id,
            "naa_monitor": state.naa_monitor_source_id,
            "dosimetry": state.dosimetry_source_id,
            "activation": state.activation_catalog_source_id,
        }[category]
        return get_nuclear_data_source(selected_id, custom_paths=self._custom_paths())

    def summary_for_category(
        self,
        category: str,
        *,
        standard: str | None = None,
    ) -> str:
        record = self.record_for_category(category, standard=standard)
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

    def apply_state(
        self,
        state: DataLibraryState | dict[str, object],
    ) -> DataLibraryState:
        """Replace the current library-selection state from a serialized payload."""

        if isinstance(state, DataLibraryState):
            next_state = state
        else:
            next_state = DataLibraryState(
                gamma_identification_source_id=str(
                    state.get(
                        "gamma_identification_source_id",
                        self._state.gamma_identification_source_id,
                    )
                ),
                calibration_source_id=str(
                    state.get("calibration_source_id", self._state.calibration_source_id)
                ),
                naa_monitor_source_id=str(
                    state.get("naa_monitor_source_id", self._state.naa_monitor_source_id)
                ),
                dosimetry_source_id=str(
                    state.get("dosimetry_source_id", self._state.dosimetry_source_id)
                ),
                activation_catalog_source_id=str(
                    state.get(
                        "activation_catalog_source_id",
                        self._state.activation_catalog_source_id,
                    )
                ),
                custom_gamma_path=self._normalize_optional_text(
                    state.get("custom_gamma_path", self._state.custom_gamma_path)
                ),
            )
        return self._publish(next_state)

    def registered_user_gamma_sources(self) -> tuple[NuclearDataSourceRecord, ...]:
        return list_registered_user_gamma_sources()

    def register_user_gamma_source(
        self,
        alias: str,
        locator: str,
        *,
        description: str | None = None,
    ) -> NuclearDataSourceRecord:
        record = register_user_gamma_source(alias, locator, description=description)
        self.set_gamma_identification_source(record.source_id)
        return record

    def remove_user_gamma_source(self, source_id: str) -> bool:
        removed = remove_user_gamma_source(source_id)
        if not removed:
            return False
        if self._state.gamma_identification_source_id == source_id:
            self.set_gamma_identification_source(
                DataLibraryState.gamma_identification_source_id
            )
        else:
            self._publish(self._state)
        return True

    def _custom_paths(self) -> tuple[str, ...]:
        return (self._state.custom_gamma_path,) if self._state.custom_gamma_path else ()


__all__ = ["DataLibraryManager", "DataLibraryState", "STANDARD_LIBRARY_LOCKS"]
