"""Named GUI workflow presets with persistent storage."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


@dataclass(frozen=True)
class WorkflowPreset:
    """One saved GUI workflow snapshot."""

    name: str
    description: str
    payload: dict[str, Any]
    built_in: bool = False


WorkflowPresetListener = Callable[[tuple[WorkflowPreset, ...], Optional[str]], None]


def _default_quantumgold_payload() -> dict[str, Any]:
    return {
        "version": 2,
        "mode_state": {
            "mode": "expert",
            "standard": None,
            "theme": "dark",
            "theme_profile": "night-lab",
        },
        "library_state": {
            "gamma_identification_source_id": "fluxforge_bundled_gamma",
            "calibration_source_id": "calibration_standard_sources",
            "naa_monitor_source_id": "k0_naa_monitors",
            "dosimetry_source_id": "irdff_ii_dosimetry",
            "activation_catalog_source_id": "flux_wire_catalog",
            "custom_gamma_path": None,
        },
        "view_state": {
            "log_scale": False,
            "peak_labels": True,
        },
        "central_state": {
            "current_tab": "Spectrum",
        },
        "workspace_state": {
            "peak_search_method": "mariscotti",
            "bayesian_source_id": "fluxforge_bundled_gamma",
            "ml_source_id": "fluxforge_bundled_gamma",
            "roi_background_method": "roi_sideband",
            "background_mode": "simple",
            "background_scale": 1.0,
            "background_visible": True,
        },
        "sidebar_state": {
            "nuclide_query": "cs",
            "nuclide_age_days": 0.0,
            "reference_tab": "Details",
            "saved_nuclides": [],
            "mixture_entries": [],
        },
        "bottom_state": {
            "current_tab": "Peak Table",
            "activity_results": {
                "background_mode": "simple",
                "background_scale": 1.0,
                "background_visible": True,
                "source_age_hours": 0.0,
                "activity_unit": "Bq",
            },
            "inventory_timeline": {
                "decay_source_id": "icrp107",
                "time_origin": "eoi",
                "observable": "activity",
                "time_start_hours": 0.0,
                "time_stop_hours": 48.0,
                "time_point_count": 25,
                "distance_cm": 30.0,
                "top_n": 8,
                "activity_unit": "Bq",
                "fim_objective": "fim-d",
                "mwdcs_window_count": 3,
                "mwdcs_full_spectrum": False,
                "advanced_objectives": False,
                "stbdmr_differentiable": False,
            },
            "masking_review": {
                "energy_window_keV": 3.0,
                "isotopes_of_interest": "",
                "top_n": 50,
                "current_tab": "Line Interference",
            },
            "optimization_workspace": {
                "objective": "di-fom",
                "irradiation_grid_s": "1800,3600,7200,14400",
                "cooldown_grid_s": "0,1800,7200,21600",
                "count_grid_s": "300,600,900,1800",
                "target_nuclide": "",
                "advanced_objectives": False,
                "current_tab": "Heatmap",
            },
            "second_irradiation": {
                "first_cooling_s": 3600.0,
                "second_irradiation_s": 1800.0,
                "flux_scales": "0.75,1.0,1.35",
                "duration_factors": "1.0,1.1,1.2",
                "cooling_grid_s": "600,900,1200",
                "target_weights": "",
            },
        },
    }


def _default_astm_ldrd_payload() -> dict[str, Any]:
    return {
        "version": 2,
        "mode_state": {
            "mode": "standards",
            "standard": "ASTM E261",
            "theme": "dark",
            "theme_profile": "night-lab",
        },
        "library_state": {
            "gamma_identification_source_id": "decay_2012",
            "calibration_source_id": "calibration_standard_sources",
            "naa_monitor_source_id": "k0_naa_monitors",
            "dosimetry_source_id": "irdff_ii_dosimetry",
            "activation_catalog_source_id": "flux_wire_catalog",
            "custom_gamma_path": None,
        },
        "view_state": {
            "log_scale": True,
            "peak_labels": True,
        },
        "central_state": {
            "current_tab": "Spectrum",
        },
        "workspace_state": {
            "peak_search_method": "mariscotti",
            "bayesian_source_id": "decay_2012",
            "ml_source_id": "decay_2012",
            "roi_background_method": "roi_sideband",
            "background_mode": "scaled",
            "background_scale": 1.0,
            "background_visible": True,
        },
        "sidebar_state": {
            "nuclide_query": "co",
            "nuclide_age_days": 0.0,
            "reference_tab": "Details",
            "saved_nuclides": [],
            "mixture_entries": [],
        },
        "bottom_state": {
            "current_tab": "Inventory / Time Evolution",
            "activity_results": {
                "background_mode": "scaled",
                "background_scale": 1.0,
                "background_visible": True,
                "source_age_hours": 24.0,
                "activity_unit": "Bq",
            },
            "inventory_timeline": {
                "decay_source_id": "icrp107",
                "time_origin": "eoi",
                "observable": "dose",
                "time_start_hours": 0.0,
                "time_stop_hours": 168.0,
                "time_point_count": 25,
                "distance_cm": 30.0,
                "top_n": 8,
                "activity_unit": "Bq",
                "fim_objective": "fim-d",
                "mwdcs_window_count": 3,
                "mwdcs_full_spectrum": True,
                "advanced_objectives": False,
                "stbdmr_differentiable": False,
            },
            "masking_review": {
                "energy_window_keV": 10.0,
                "isotopes_of_interest": "",
                "top_n": 50,
                "current_tab": "Line Interference",
            },
            "optimization_workspace": {
                "objective": "di-fom",
                "irradiation_grid_s": "1800,3600,7200,14400",
                "cooldown_grid_s": "0,1800,7200,21600",
                "count_grid_s": "300,600,900,1800",
                "target_nuclide": "",
                "advanced_objectives": False,
                "current_tab": "Recommendation",
            },
            "second_irradiation": {
                "first_cooling_s": 3600.0,
                "second_irradiation_s": 1800.0,
                "flux_scales": "0.75,1.0,1.35",
                "duration_factors": "1.0,1.1,1.2",
                "cooling_grid_s": "600,900,1200",
                "target_weights": "",
            },
        },
    }


DEFAULT_WORKFLOW_PRESETS: dict[str, dict[str, Any]] = {
    "quantumgold-workflow": {
        "description": (
            "Peak-centric Expert-mode workspace aligned with QuantumGold-style "
            "manual review, background inspection, and library-backed peak work."
        ),
        "payload": _default_quantumgold_payload(),
    },
    "astm-ldrd-irradiation": {
        "description": (
            "Standards-locked ASTM E261 workspace tuned for the LDRD irradiation "
            "analysis path, inventory review, and follow-on optimization surfaces."
        ),
        "payload": _default_astm_ldrd_payload(),
    },
}


class WorkflowPresetManager:
    """Persist named workflow presets across GUI sessions."""

    WORKFLOW_PRESETS_KEY = "gui/workflow_presets"
    ACTIVE_WORKFLOW_KEY = "gui/active_workflow"

    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._user_presets = self._load_user_presets()
        self._active_workflow = self._load_active_workflow()
        self._listeners: list[WorkflowPresetListener] = []

    def subscribe(self, listener: WorkflowPresetListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: WorkflowPresetListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def available_workflows(self) -> tuple[WorkflowPreset, ...]:
        presets: list[WorkflowPreset] = []
        for name, item in DEFAULT_WORKFLOW_PRESETS.items():
            presets.append(
                WorkflowPreset(
                    name=name,
                    description=str(item.get("description") or ""),
                    payload=copy.deepcopy(dict(item.get("payload") or {})),
                    built_in=True,
                )
            )
        for name in sorted(self._user_presets):
            item = self._user_presets[name]
            presets.append(
                WorkflowPreset(
                    name=name,
                    description=str(item.get("description") or ""),
                    payload=copy.deepcopy(dict(item.get("payload") or {})),
                    built_in=False,
                )
            )
        return tuple(presets)

    def workflow_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.available_workflows())

    def get_workflow(self, name: str) -> WorkflowPreset | None:
        workflow_name = str(name or "").strip()
        if not workflow_name:
            return None
        for item in self.available_workflows():
            if item.name == workflow_name:
                return item
        return None

    def active_workflow_name(self) -> str | None:
        name = str(self._active_workflow or "").strip()
        return name or None

    def active_workflow(self) -> WorkflowPreset | None:
        name = self.active_workflow_name()
        if not name:
            return None
        return self.get_workflow(name)

    def mode_state_for(self, name: str | None = None) -> dict[str, Any]:
        """Return the mode-state payload for a named or active workflow."""

        preset = self._resolve_workflow(name)
        if preset is None:
            return {}
        mode_state, _library_state = self.extract_mode_and_library_state(preset.payload)
        return mode_state

    def library_state_for(self, name: str | None = None) -> dict[str, Any]:
        """Return the library-state payload for a named or active workflow."""

        preset = self._resolve_workflow(name)
        if preset is None:
            return {}
        _mode_state, library_state = self.extract_mode_and_library_state(preset.payload)
        return library_state

    def apply_mode_and_library_state(
        self,
        *,
        name: str | None = None,
        mode_state_applier: Callable[[Mapping[str, Any]], None] | None = None,
        library_state_applier: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> WorkflowPreset | None:
        """Apply mode/library slices for a named or active workflow preset."""

        preset = self._resolve_workflow(name)
        if preset is None:
            return None
        mode_state, library_state = self.extract_mode_and_library_state(preset.payload)
        if mode_state and callable(mode_state_applier):
            mode_state_applier(mode_state)
        if library_state and callable(library_state_applier):
            library_state_applier(library_state)
        return preset

    @staticmethod
    def extract_mode_and_library_state(
        payload: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract mode/library state dictionaries from a workflow payload."""

        if not isinstance(payload, Mapping):
            return {}, {}
        mode_payload = payload.get("mode_state")
        library_payload = payload.get("library_state")
        mode_state = (
            copy.deepcopy(dict(mode_payload))
            if isinstance(mode_payload, Mapping)
            else {}
        )
        library_state = (
            copy.deepcopy(dict(library_payload))
            if isinstance(library_payload, Mapping)
            else {}
        )
        return mode_state, library_state

    def set_active_workflow(self, name: str | None) -> str | None:
        workflow_name = str(name or "").strip() or None
        if workflow_name is not None and self.get_workflow(workflow_name) is None:
            raise ValueError(f"Unknown workflow preset {workflow_name!r}")
        self._active_workflow = workflow_name
        self._save_active_workflow()
        self._publish()
        return self._active_workflow

    def save_workflow(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        description: str | None = None,
    ) -> WorkflowPreset:
        workflow_name = str(name or "").strip()
        if not workflow_name:
            raise ValueError("Workflow name cannot be empty.")
        if workflow_name in DEFAULT_WORKFLOW_PRESETS:
            raise ValueError(
                f"{workflow_name!r} is a built-in workflow preset and cannot be overwritten."
            )
        self._user_presets[workflow_name] = {
            "description": str(description or "").strip(),
            "payload": copy.deepcopy(dict(payload)),
        }
        self._save_user_presets()
        self._active_workflow = workflow_name
        self._save_active_workflow()
        self._publish()
        saved = self.get_workflow(workflow_name)
        if saved is None:  # pragma: no cover - defensive only
            raise RuntimeError("Workflow preset save did not round-trip.")
        return saved

    def delete_workflow(self, name: str) -> bool:
        workflow_name = str(name or "").strip()
        if not workflow_name or workflow_name in DEFAULT_WORKFLOW_PRESETS:
            return False
        removed = self._user_presets.pop(workflow_name, None) is not None
        if not removed:
            return False
        self._save_user_presets()
        if self._active_workflow == workflow_name:
            self._active_workflow = None
            self._save_active_workflow()
        self._publish()
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "active_workflow": self.active_workflow_name(),
            "workflow_names": list(self.workflow_names()),
            "built_in_workflows": list(DEFAULT_WORKFLOW_PRESETS.keys()),
        }

    def _publish(self) -> None:
        presets = self.available_workflows()
        active_name = self.active_workflow_name()
        for listener in tuple(self._listeners):
            listener(presets, active_name)

    def _load_user_presets(self) -> dict[str, dict[str, Any]]:
        raw = self._settings_value(self.WORKFLOW_PRESETS_KEY, "")
        if not raw:
            return {}
        try:
            parsed = json.loads(str(raw))
        except (TypeError, json.JSONDecodeError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        normalized: dict[str, dict[str, Any]] = {}
        for key, value in parsed.items():
            name = str(key or "").strip()
            if not name or name in DEFAULT_WORKFLOW_PRESETS:
                continue
            if not isinstance(value, dict):
                continue
            payload = value.get("payload")
            if not isinstance(payload, dict):
                continue
            normalized[name] = {
                "description": str(value.get("description") or "").strip(),
                "payload": copy.deepcopy(payload),
            }
        return normalized

    def _save_user_presets(self) -> None:
        self._settings_set_value(
            self.WORKFLOW_PRESETS_KEY,
            json.dumps(self._user_presets, sort_keys=True),
        )
        self._sync_settings()

    def _load_active_workflow(self) -> str | None:
        value = self._settings_value(self.ACTIVE_WORKFLOW_KEY, "")
        name = str(value or "").strip()
        if not name:
            return None
        if name in DEFAULT_WORKFLOW_PRESETS or name in self._user_presets:
            return name
        return None

    def _resolve_workflow(self, name: str | None = None) -> WorkflowPreset | None:
        resolved_name = str(name or "").strip()
        if resolved_name:
            return self.get_workflow(resolved_name)
        return self.active_workflow()

    def _save_active_workflow(self) -> None:
        self._settings_set_value(self.ACTIVE_WORKFLOW_KEY, self._active_workflow or "")
        self._sync_settings()

    def _settings_value(self, key: str, default):
        getter = getattr(self._settings, "value", None)
        if callable(getter):
            return getter(key, default)
        return default

    def _settings_set_value(self, key: str, value) -> None:
        setter = getattr(self._settings, "setValue", None)
        if callable(setter):
            setter(key, value)

    def _sync_settings(self) -> None:
        sync = getattr(self._settings, "sync", None)
        if callable(sync):
            sync()


__all__ = [
    "DEFAULT_WORKFLOW_PRESETS",
    "WorkflowPreset",
    "WorkflowPresetManager",
]
