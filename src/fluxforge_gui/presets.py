"""Standards presets and selector helpers for FluxForge GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fluxforge.data.nuclear_data_sources import list_nuclear_data_sources
from fluxforge.data.rafm_profile import list_rafm_profiles
from fluxforge_gui.models import StandardsGuiPreset


def get_gui_profile_choices() -> tuple[str, ...]:
    """Return bundled detector profile choices for GUI selectors."""

    return ("", *list_rafm_profiles())


def get_gui_data_source_choices(
    custom_paths: Iterable[str | Path] = (),
) -> tuple[str, ...]:
    """Return selectable nuclear data source identifiers for the GUI."""

    return tuple(record.source_id for record in list_nuclear_data_sources(custom_paths))


def get_standards_gui_presets() -> dict[str, StandardsGuiPreset]:
    """Return standards workflow presets surfaced by the GUI."""

    return {
        "astm_inl": StandardsGuiPreset(
            key="astm_inl",
            label="ASTM / INL dosimetry",
            default_profile="astm_inl_dosimetry",
            peaks_sensitivity="conservative",
            peak_counting_method="iec_tiered",
            background_subtracted=True,
            reaction_category="fast",
            notes=(
                "Applies the bundled ASTM/INL detector profile to ingest and "
                "manual-ROI peak workflows.",
                "Use measured background subtraction, manual ROI files, and "
                "IEC-tiered counting for standards-style peak review.",
                "Pairs well with IRDFF-II fast-reaction browsing for monitor "
                "selection.",
            ),
        ),
        "us_astm": StandardsGuiPreset(
            key="us_astm",
            label="US ASTM reactor dosimetry",
            default_profile="us_astm_reactor_dosimetry",
            peaks_sensitivity="conservative",
            peak_counting_method="iec_tiered",
            background_subtracted=True,
            reaction_category="fast",
            notes=(
                "Uses the US ASTM reactor-dosimetry bundled profile for "
                "consistent background and efficiency defaults.",
                "Designed for standards-driven dosimetry spectra with "
                "explicit ROI review.",
                "Use the reactions browser to narrow candidate threshold "
                "reactions by target or category.",
            ),
        ),
        "iaea_irdff_gma": StandardsGuiPreset(
            key="iaea_irdff_gma",
            label="IAEA IRDFF / GMA dosimetry",
            default_profile="astm_inl_dosimetry",
            peaks_sensitivity="conservative",
            peak_counting_method="iec_tiered",
            background_subtracted=True,
            reaction_category="all",
            notes=(
                "Targets the IAEA-style dosimetry path backed by IRDFF-II "
                "reactions, response-matrix construction, and GMA-like "
                "advanced solvers.",
                "Use the Unfold tab with IRDFF-derived responses and the "
                "Standards reactions browser to inspect thermal, epithermal, "
                "fast, and fission monitors.",
                "This preset reuses the bundled dosimetry detector profile "
                "where a practical GUI default is needed, but the physics "
                "backend is the IAEA IRDFF/GMA workflow.",
            ),
        ),
        "k0_naa": StandardsGuiPreset(
            key="k0_naa",
            label="k0-NAA",
            default_profile=None,
            peaks_sensitivity="sensitive",
            peak_counting_method="covell_local",
            background_subtracted=True,
            reaction_category="thermal",
            notes=(
                "Configures the GUI for k0-NAA style peak review with manual "
                "ROI support and background-subtracted integration.",
                "Enter timing, efficiency, isotope, and comparator-specific "
                "values on the Activity tab after peak extraction.",
                "Thermal and epithermal monitor browsing is exposed through "
                "the IRDFF reactions browser.",
            ),
        ),
        "comparator_naa": StandardsGuiPreset(
            key="comparator_naa",
            label="Comparator NAA",
            default_profile=None,
            peaks_sensitivity="default",
            peak_counting_method="covell_local",
            background_subtracted=True,
            reaction_category="thermal",
            notes=(
                "Sets up manual-ROI and background-subtracted peak processing "
                "for comparator-based NAA workflows.",
                "Use Activity tab overrides to keep isotope and reaction "
                "identifiers aligned with comparator calculations.",
                "Report and Rates tabs can then bundle the resulting "
                "activities into downstream validation artifacts.",
            ),
        ),
        "physics_like": StandardsGuiPreset(
            key="physics_like",
            label="Physics-style activation / spectroscopy",
            default_profile=None,
            peaks_sensitivity="default",
            peak_counting_method="gaussian_fit",
            background_subtracted=True,
            reaction_category="all",
            notes=(
                "Maps to FluxForge modules that parallel Physics-style "
                "workflows: Bateman decay chains, stacked-target energy "
                "degradation, XCOM attenuation, dose-rate estimates, and SPE "
                "export.",
                "Use the Peaks, Activity, and Rates tabs for HPGe-style "
                "spectroscopy, then pair with stacked-target and activation "
                "modules from the Python API for charged-particle workflows.",
                "This preset is informational for the GUI today because the "
                "stacked-target and decay-chain parity modules are "
                "implemented in code and tests but are not yet full GUI tasks.",
            ),
        ),
        "curie_like": StandardsGuiPreset(
            key="curie_like",
            label="Curie-style activation / spectroscopy",
            default_profile=None,
            peaks_sensitivity="default",
            peak_counting_method="gaussian_fit",
            background_subtracted=True,
            reaction_category="all",
            notes=(
                "Maps to FluxForge modules that parallel Curie-style "
                "workflows: Bateman decay chains, stacked-target energy "
                "degradation, XCOM attenuation, dose-rate estimates, and SPE "
                "export.",
                "Use the Peaks, Activity, and Rates tabs for HPGe-style "
                "spectroscopy, then pair with stacked-target and activation "
                "modules from the Python API for charged-particle workflows.",
                "This preset is informational for the GUI today because the "
                "stacked-target and decay-chain parity modules are "
                "implemented in code and tests but are not yet full GUI tasks.",
            ),
        ),
    }


def build_standards_preset_values(
    preset_key: str,
    selected_profile: str | None = None,
) -> dict[str, str | bool]:
    """Translate a standards preset into GUI field values."""

    preset = get_standards_gui_presets()[preset_key]
    profile = (selected_profile or "").strip() or (preset.default_profile or "")
    return {
        "profile": profile,
        "peaks_sensitivity": preset.peaks_sensitivity,
        "peak_counting_method": preset.peak_counting_method,
        "background_subtracted": preset.background_subtracted,
        "reaction_category": preset.reaction_category,
        "notes": "\n".join(f"• {line}" for line in preset.notes),
    }
