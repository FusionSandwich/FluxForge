import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluxforge.examples.rafm_workflow import (
    default_paths,
    qg_reference_peaks,
    workflow_profile_energy_calibration,
)
from fluxforge.io.flux_wire import read_processed_txt, read_raw_asc
from examples.RAFM_irradiation.compare_peak_count_methods import (
    compute_current_method_results,
)


def test_qg_benchmark_predictions_match_reference_for_representative_ti_sample():
    example_root = Path(__file__).resolve().parents[1] / "examples" / "RAFM_irradiation"
    paths = default_paths(example_root)
    config = json.loads(
        (paths.metadata_root / "workflow_config.json").read_text(encoding="utf-8")
    )
    energy_override = workflow_profile_energy_calibration(config)

    sample_id = "Ti-RAFM-1a_25cm"
    raw_path = paths.raw_root / "flux_wires" / f"{sample_id}.ASC"
    qg_path = paths.qg_root / "flux_wires" / f"{sample_id}.txt"

    raw_data = read_raw_asc(
        raw_path,
        energy_calibration_override=energy_override,
        profile_name=str(config["profile_name"]),
    )
    raw_data.sample_id = sample_id
    reference_data = read_processed_txt(
        qg_path, profile_name=str(config["profile_name"])
    )
    background_data = read_raw_asc(
        paths.background_path,
        energy_calibration_override=energy_override,
        profile_name=str(config["profile_name"]),
    )

    predictions = compute_current_method_results(
        raw_data, reference_data, background_data, config
    )

    matched = 0
    for ref_peak in qg_reference_peaks(reference_data):
        if float(ref_peak["net_counts"]) < float(
            config.get("minimum_qg_net_counts", 1.0)
        ):
            continue
        key = f"{ref_peak['isotope']}@{float(ref_peak['energy_keV']):.2f}"
        prediction = predictions.get(key)
        if prediction is None:
            continue
        matched += 1
        assert prediction["net"] == float(ref_peak["net_counts"])
        assert prediction["gross"] == float(ref_peak.get("gross_counts") or 0.0)

    assert matched >= 4
