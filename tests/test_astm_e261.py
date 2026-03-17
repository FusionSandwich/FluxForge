from __future__ import annotations

import json
from pathlib import Path

from fluxforge.analysis.astm_e261 import analyze_astm_e261_plan, target_atom_count
from fluxforge.cli import app as cli_app
from fluxforge_gui.app import build_gui_astm_e261_preview


def test_target_atom_count_accounts_for_abundance_and_purity() -> None:
    atoms = target_atom_count(
        sample_mass_g=1.0,
        atomic_mass_g_mol=100.0,
        isotopic_abundance=0.5,
        mass_fraction=0.2,
        sample_purity=0.8,
        atoms_per_formula_unit=2.0,
    )

    assert atoms > 0.0
    assert 9.0e20 < atoms < 1.1e21


def test_analyze_astm_e261_plan_builds_fluence_bundle(tmp_path: Path) -> None:
    plan = {
        "schema": "fluxforge.astm_e261_plan.v1",
        "title": "ASTM E261 smoke test",
        "irradiation": {"segments": [{"duration_s": 600.0, "relative_power": 1.0}]},
        "measurements": [
            {
                "measurement_id": "co60-monitor",
                "reaction_id": "Co-59(n,g)Co-60",
                "monitor_id": "co-wire",
                "product_isotope": "Co-60",
                "target_isotope": "Co-59",
                "line_energy_keV": 1332.5,
                "net_counts": 2500.0,
                "live_time_s": 100.0,
                "efficiency": 0.01,
                "gamma_intensity": 0.9998,
                "half_life_s": 166344192.0,
                "cooling_time_s": 3600.0,
                "sample_mass_g": 0.001,
                "atomic_mass_g_mol": 58.933,
                "isotopic_abundance": 1.0,
                "effective_cross_section_barn": 37.18,
                "effective_cross_section_unc_barn": 0.5,
            }
        ],
    }

    result = analyze_astm_e261_plan(plan)

    assert result["schema"] == "fluxforge.astm_e261_result.v1"
    assert result["summary"]["measurement_count"] == 1
    monitor = result["measurements"][0]
    assert monitor["reaction_id"] == "Co-59(n,g)Co-60"
    assert monitor["activity_eoi_Bq"] > 0.0
    assert monitor["reaction_rate_s"] > 0.0
    assert monitor["target_atoms"] > 0.0
    assert monitor["fluence_rate_cm2_s"] > 0.0
    assert monitor["fluence_cm2"] > monitor["fluence_rate_cm2_s"]

    output_path = tmp_path / "astm_e261.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    preview = build_gui_astm_e261_preview(result, output_path)
    assert "FluxForge ASTM E261 Preview" in preview
    assert "Co-59(n,g)Co-60" in preview


def test_astm_e261_cli_parser_registers_command() -> None:
    parser = cli_app.build_parser()
    args = parser.parse_args(["astm-e261", "--plan-file", "plan.json", "--output", "out.json"])

    assert args.plan_file.name == "plan.json"
    assert args.output.name == "out.json"
    assert args.func is cli_app.cmd_astm_e261


def test_astm_e261_cli_command_writes_output(tmp_path: Path) -> None:
    plan = {
        "schema": "fluxforge.astm_e261_plan.v1",
        "irradiation": {"duration_s": 600.0},
        "measurements": [
            {
                "measurement_id": "m1",
                "reaction_id": "Co-59(n,g)Co-60",
                "product_isotope": "Co-60",
                "net_counts": 2000.0,
                "live_time_s": 100.0,
                "efficiency": 0.01,
                "gamma_intensity": 0.9998,
                "half_life_s": 166344192.0,
                "cooling_time_s": 3600.0,
                "sample_mass_g": 0.001,
                "atomic_mass_g_mol": 58.933,
                "isotopic_abundance": 1.0,
                "effective_cross_section_barn": 37.18,
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "out.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    parser = cli_app.build_parser()
    args = parser.parse_args(["astm-e261", "--plan-file", str(plan_path), "--output", str(output_path)])
    args.func(args)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "fluxforge.astm_e261_result.v1"
    assert payload["summary"]["measurement_count"] == 1
