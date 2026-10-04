"""Portable, bounded line-count evidence using the current validation engine.

Run from a source checkout: python examples/qg_protocol/run_example.py --output NEW.json
No activity reproduction or vendor parity is claimed. Physical background
applicability remains conditional, independently of the comparison controls.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fluxforge.analysis.flux_wire_analysis import estimate_peak_area_local_background
from fluxforge.analysis.qg_protocol import (
    HistoricalProtocol,
    LAYOUT_ID,
    parse_saved_state,
    verified_sha256,
)
from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.io.flux_wire import read_processed_txt, read_raw_asc


FIXTURES = Path(__file__).parent / "fixtures"


def audit_originals(originals: Path, manifest: dict) -> list[dict]:
    """Verify all 32 expected identities; extra or missing ANS files fail."""
    expected = {Path(row["source"]).name for row in manifest["rows"]}
    if {p.name for p in (originals / "ANS").glob("*.ANS")} != expected:
        raise ValueError("Original ANS set differs from the 32-file source manifest")
    rows = []
    for row in manifest["rows"]:
        report = (
            (originals / row["report_source"]).read_bytes()
            if row["report_sha256"]
            else None
        )
        state = parse_saved_state(
            (originals / row["source"]).read_bytes(),
            expected_sha256=row["sha256"],
            layout_id=LAYOUT_ID,
            report=report,
            expected_report_sha256=row["report_sha256"],
        )
        if state.values != row["expected_values"] or state.byte_length != row["bytes"]:
            raise ValueError("Saved settings differ from source-bound investigation")
        rows.append({"source": row["source"], **state.to_dict()})
    return rows


def line_counts(spectrum, channel, fwhm, protocol, *, continuum):
    params = protocol.roi_parameters() if protocol is not None else {}
    net, unc, gross, local, bounds = estimate_peak_area_local_background(
        spectrum.counts,
        channel,
        fwhm,
        spectrum_data=spectrum,
        **params,
    )
    if not continuum:
        weights = np.zeros(len(spectrum.counts))
        weights[bounds[0] : bounds[1] + 1] = 1.0
        net = gross
        unc = float(np.sqrt(spectrum.weighted_counts_variance(weights)))
        local = 0.0
    return {
        "net_counts": net,
        "net_counts_unc": unc,
        "gross_counts": gross,
        "local_continuum_counts": local,
        "roi_channels": list(bounds),
        "continuum_enabled": continuum,
        "method": (
            "current_engine_local_sidebands"
            if continuum
            else "current_engine_roi_gross"
        ),
    }


def build_evidence(
    *,
    ambient: str | None = None,
    continuum: str | None = None,
    originals: Path = FIXTURES,
    aggregation: str | None = None,
) -> dict:
    if ambient not in {None, "on", "off"} or continuum not in {None, "on", "off"}:
        raise ValueError("Correction choices must be on/off or omitted")
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    # Bind the reused implementation, not merely a claimed Git revision.
    for path, digest in manifest["engine_files"].items():
        verified_sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n"), digest)
    audit = audit_originals(originals, manifest)
    row = next(r for r in manifest["rows"] if r["source"] == "ANS/Co-Cd-RAFM-1.ANS")
    state = parse_saved_state(
        (originals / row["source"]).read_bytes(),
        expected_sha256=row["sha256"],
        layout_id=LAYOUT_ID,
        report=(originals / row["report_source"]).read_bytes(),
        expected_report_sha256=row["report_sha256"],
    )
    protocol = HistoricalProtocol.from_saved_state(
        state, scenario_name="saved_header_comparison"
    )
    choices = {"continuum_method": "current_engine_local_sidebands"}
    if ambient is not None:
        choices["ambient_enabled"] = ambient == "on"
    if continuum is not None:
        choices["continuum_enabled"] = continuum == "on"
    if aggregation is not None:
        choices["aggregation"] = aggregation
    protocol = protocol.with_assumptions(
        scenario_name="declared_header_reproduction",
        rationale="Explicit current-engine comparison; exact vendor continuum/aggregation not established",
        **choices,
    )
    asc = FIXTURES / manifest["focused_asc"]["source"]
    verified_sha256(asc.read_bytes(), manifest["focused_asc"]["sha256"])
    report_data = read_processed_txt(originals / row["report_source"])
    raw = read_raw_asc(asc, energy_calibration_override=report_data.energy_calibration)
    sample = raw.spectrum
    background = ROOT / manifest["physical_background"]["source"]
    verified_sha256(background.read_bytes(), manifest["physical_background"]["sha256"])
    measured = read_raw_asc(background).spectrum
    physical_spectrum = subtract_measured_background(sample, measured, mode="live")
    physical_basis = {
        **manifest["physical_background"],
        "sample_sha256": manifest["focused_asc"]["sha256"],
        "energy_calibration_source_sha256": row["report_sha256"],
        "normalization": "live_time",
        "scale_factor": float(sample.live_time / measured.live_time),
        "ambient_enabled": True,
        "continuum_enabled": True,
        "roi_parameters": {
            "roi_width_fwhm": 4.0,
            "background_width_channels": 1,
            "background_gap_fwhm": 0.0,
        },
        "roi_parameter_source": "Unchanged current engine defaults, independent of comparison protocol",
    }
    enabled = protocol.fields["ambient_enabled"].value
    comparison_spectrum = physical_spectrum if enabled else sample
    if enabled:
        protocol = protocol.with_assumptions(
            scenario_name=protocol.scenario_name,
            rationale="Explicit conditional North comparison scenario; applicability unestablished",
            ambient_identity=physical_basis["source"],
            ambient_sha256=physical_basis["sha256"],
            ambient_normalization="live_time",
        )
    comparison_basis = {
        "sample_sha256": manifest["focused_asc"]["sha256"],
        "energy_calibration_source_sha256": row["report_sha256"],
        "ambient_enabled": enabled,
        "ambient_source_sha256": physical_basis["sha256"] if enabled else None,
        "ambient_normalization": "live_time" if enabled else "none",
        "scale_factor": physical_basis["scale_factor"] if enabled else None,
        "qualification": "Comparison only; saved controls do not establish final report controls",
    }
    lines = []
    # Fixed declared Co-60 lines, never tuned to reported activities. Resolution
    # uses the printed report polynomial (including its quadratic term).
    for energy in (1173.2, 1332.5):
        channel = int(np.argmin(abs(sample.energies - energy)))
        resolution_keV = 1.389 + 7.800e-4 * energy - 4.072e-8 * energy**2
        slope = (
            report_data.energy_calibration[1]
            + 2 * report_data.energy_calibration[2] * channel
        )
        fwhm = resolution_keV / slope
        lines.append(
            {
                "energy_keV": energy,
                "channel": channel,
                "fwhm_channels": fwhm,
                "resolution_source_sha256": row["report_sha256"],
                "physical_counts": line_counts(
                    physical_spectrum, channel, fwhm, None, continuum=True
                ),
                "comparison_counts": line_counts(
                    comparison_spectrum,
                    channel,
                    fwhm,
                    protocol,
                    continuum=protocol.fields["continuum_enabled"].value,
                ),
            }
        )
    return {
        "status": "BOUNDED_COMPONENT_EVIDENCE; workflow integration pending #232/#220",
        "engine_base_revision": manifest["engine_base_revision"],
        "engine_file_sha256": manifest["engine_files"],
        "engine_hash_basis": "canonical_lf_source_bytes",
        "component_sha256": hashlib.sha256(
            (ROOT / "src/fluxforge/analysis/qg_protocol.py").read_bytes()
        ).hexdigest(),
        "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": {
            "python": platform.python_version(),
            **{n: importlib.metadata.version(n) for n in ("numpy", "scipy")},
        },
        "protocol": protocol.to_dict(),
        "saved_headers": audit,
        "physical_count_basis": physical_basis,
        "comparison_count_basis": comparison_basis,
        "line_evidence": lines,
        "activity_computation": "Not performed; original GammaLib contents/corrections unknown",
        "aggregation_computation": "Not performed; configured choice is metadata only",
        "exact_vendor_parity": False,
        "physical_defaults_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON; no overwrite"
    )
    parser.add_argument(
        "--ambient", choices=("on", "off"), help="Comparison scenario only"
    )
    parser.add_argument(
        "--continuum", choices=("on", "off"), help="Independent comparison control"
    )
    parser.add_argument(
        "--aggregation", help="Persist a choice; no aggregation is executed"
    )
    parser.add_argument(
        "--originals", type=Path, default=FIXTURES, help="Root with ANS and QG_report"
    )
    args = parser.parse_args()
    evidence = build_evidence(
        ambient=args.ambient,
        continuum=args.continuum,
        originals=args.originals,
        aggregation=args.aggregation,
    )
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(evidence, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "saved_headers": len(evidence["saved_headers"]),
                "lines": len(evidence["line_evidence"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
