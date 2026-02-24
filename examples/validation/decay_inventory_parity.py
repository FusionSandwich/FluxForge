import csv
import json
import re
from pathlib import Path

from fluxforge.physics.decay_library import DecayDataset
from fluxforge.physics.decay_inventory import DecayInventory


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["nuclide", "value"])
        writer.writerows(rows)


def _parse_flux_wire_activities(report_path: Path) -> dict:
    text = report_path.read_text(encoding="latin-1", errors="ignore")
    lines = text.splitlines()
    activities = {}
    table_start = None
    for idx, line in enumerate(lines):
        if "Activity" in line and "ROI" in line:
            table_start = idx + 2
            break
    if table_start is None:
        return activities

    activity_re = re.compile(r"([A-Za-z0-9]+)@")
    for line in lines[table_start:]:
        if not line.strip():
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        match = activity_re.search(line)
        if not match:
            continue
        nuclide_raw = match.group(1)
        try:
            activity_uci = float(parts[-1].replace(",", ""))
        except ValueError:
            continue
        nuclide = f"{nuclide_raw[:-2]}-{nuclide_raw[-2:]}" if nuclide_raw[-2:].isdigit() else nuclide_raw
        activities.setdefault(nuclide, []).append(activity_uci)

    return {k: sum(v) / len(v) for k, v in activities.items()}


def main() -> None:
    output_dir = Path("FluxForge/artifacts/validation/decay_inventory")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(
        "testing/radioactivedecay/radioactivedecay/icrp107_ame2020_nubase2020/decay_data.npz"
    )
    dataset = DecayDataset.from_radioactivedecay_npz(dataset_path)

    # Reference parity dataset (radioactivedecay README)
    inventory = DecayInventory.from_quantities({"Mo-99": 2.0}, unit="bq", dataset=dataset)
    decayed = inventory.decay(20.0, units="h")
    activities = decayed.activities("bq")
    cumulative = inventory.cumulative_decays(20.0, units="h")

    _write_json(output_dir / "mo99_activity.json", {"activities_bq": activities})
    _write_csv(output_dir / "mo99_activity.csv", activities.items())
    _write_json(output_dir / "mo99_cumulative.json", {"decays": cumulative})
    _write_csv(output_dir / "mo99_cumulative.csv", cumulative.items())

    # UWNR regression dataset: processed flux wire report
    report_path = Path(
        "rafm_irradiation_ldrd/irradiation_QG_processed/flux_wires/Co-Cd-RAFM-1_25cm.txt"
    )
    uci = _parse_flux_wire_activities(report_path)
    bq = {k: v * 1e-6 * 3.7e10 for k, v in uci.items()}

    if bq:
        inv_flux = DecayInventory.from_quantities(bq, unit="bq", dataset=dataset)
        decayed_flux = inv_flux.decay(7.0, units="d")
        _write_json(
            output_dir / "rafm_co60_activity.json",
            {"activities_bq": decayed_flux.activities("bq")},
        )
        _write_csv(output_dir / "rafm_co60_activity.csv", decayed_flux.activities("bq").items())


if __name__ == "__main__":
    main()
