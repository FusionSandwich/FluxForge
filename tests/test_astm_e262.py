from __future__ import annotations

import json
from pathlib import Path

from fluxforge.analysis.astm_e262 import analyze_astm_e262_plan
from fluxforge.cli import app as cli_app
from fluxforge_gui.app import build_gui_astm_e262_preview


def test_analyze_astm_e262_plan_radiometric_with_cd_pair(tmp_path: Path) -> None:
    plan = {
        "schema": "fluxforge.astm_e262_plan.v1",
        "title": "ASTM E262 smoke test",
        "irradiation": {"segments": [{"duration_s": 900.0, "relative_power": 1.0}]},
        "measurements": [
            {
                "measurement_id": "au-monitor",
                "reaction_id": "Au-197(n,g)Au-198",
                "mode": "radiometric",
                "net_counts": 8000.0,
                "live_time_s": 200.0,
                "efficiency": 0.01,
                "gamma_intensity": 0.955,
                "half_life_s": 232848.0,
                "cooling_time_s": 1200.0,
                "cd_net_counts": 600.0,
                "cd_live_time_s": 200.0,
                "cd_efficiency": 0.01,
                "cd_gamma_intensity": 0.955,
                "cd_half_life_s": 232848.0,
                "cd_cooling_time_s": 1200.0,
                "sigma_0_barn": 98.65,
                "sigma_0_unc_barn": 0.5,
                "westcott_g": 1.0,
                "thermal_self_shielding_factor": 0.99,
                "cd_thickness_mm": 1.0,
            }
        ],
    }

    result = analyze_astm_e262_plan(plan)

    assert result["schema"] == "fluxforge.astm_e262_result.v1"
    assert result["summary"]["measurement_count"] == 1
    row = result["measurements"][0]
    assert row["mode"] == "radiometric"
    assert row["equivalent_2200ms_fluence_rate_cm2_s"] > 0.0
    assert (
        row["equivalent_2200ms_fluence_cm2"]
        > row["equivalent_2200ms_fluence_rate_cm2_s"]
    )
    assert "cadmium_ratio" in row

    output_path = tmp_path / "astm_e262.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    preview = build_gui_astm_e262_preview(result, output_path)
    assert "FluxForge ASTM E262 Preview" in preview
    assert "Au-197(n,g)Au-198" in preview


def test_analyze_astm_e262_plan_standard_comparison_mode() -> None:
    plan = {
        "irradiation": {"duration_s": 600.0},
        "measurements": [
            {
                "measurement_id": "gold-standard-comparison",
                "reaction_id": "Au-197(n,g)Au-198",
                "mode": "standard_comparison",
                "known_reference_fluence_rate_cm2_s": 1000000.0,
                "known_reference_fluence_rate_unc_cm2_s": 10000.0,
                "spectral_correction_factor": 0.98,
                "unknown_net_counts": 5000.0,
                "unknown_live_time_s": 120.0,
                "unknown_efficiency": 0.012,
                "unknown_gamma_intensity": 0.955,
                "unknown_half_life_s": 232848.0,
                "unknown_cooling_time_s": 600.0,
                "standard_net_counts": 5200.0,
                "standard_live_time_s": 120.0,
                "standard_efficiency": 0.012,
                "standard_gamma_intensity": 0.955,
                "standard_half_life_s": 232848.0,
                "standard_cooling_time_s": 600.0,
            }
        ],
    }

    result = analyze_astm_e262_plan(plan)
    row = result["measurements"][0]
    assert row["mode"] == "standard_comparison"
    assert row["equivalent_2200ms_fluence_rate_cm2_s"] > 0.0


def test_astm_e262_cli_parser_registers_command() -> None:
    parser = cli_app.build_parser()
    args = parser.parse_args(
        ["astm-e262", "--plan-file", "plan.json", "--output", "out.json"]
    )

    assert args.plan_file.name == "plan.json"
    assert args.output.name == "out.json"
    assert args.func is cli_app.cmd_astm_e262


def test_astm_e262_cli_command_writes_output(tmp_path: Path) -> None:
    plan = {
        "schema": "fluxforge.astm_e262_plan.v1",
        "irradiation": {"duration_s": 600.0},
        "measurements": [
            {
                "measurement_id": "m1",
                "reaction_id": "Co-59(n,g)Co-60",
                "mode": "radiometric",
                "net_counts": 2400.0,
                "live_time_s": 100.0,
                "efficiency": 0.01,
                "gamma_intensity": 0.9998,
                "half_life_s": 166344192.0,
                "cooling_time_s": 3600.0,
                "sigma_0_barn": 37.18,
                "westcott_g": 1.0,
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "out.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    parser = cli_app.build_parser()
    args = parser.parse_args(
        ["astm-e262", "--plan-file", str(plan_path), "--output", str(output_path)]
    )
    args.func(args)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.astm_e262_result.v1"
    assert payload["summary"]["measurement_count"] == 1


def test_analyze_astm_e262_plan_fallback_to_library() -> None:
    # Omit sigma_0_barn, it should resolve from Co-60 via k0 library lookup
    plan = {
        "irradiation": {"duration_s": 86400.0},
        "measurements": [
            {
                "measurement_id": "Co-60",
                "reaction_id": "Co-59(n,g)Co-60",
                "mode": "radiometric",
                "net_counts": 5000.0,
                "live_time_s": 1000.0,
                "efficiency": 0.05,
                "gamma_intensity": 1.0,
                "half_life_s": 166340000.0,
                "cooling_time_s": 3600.0,
                "thermal_self_shielding_factor": 1.0,
            }
        ],
    }

    result = analyze_astm_e262_plan(plan)
    row = result["measurements"][0]

    # Check that sigma_0_barn was indeed resolved (Co-59 thermal cross-section is ~37.18 barns)
    assert "sigma_0_barn" in row
    assert row["sigma_0_barn"] > 30.0 and row["sigma_0_barn"] < 45.0
    assert row["equivalent_2200ms_fluence_rate_cm2_s"] > 0.0
