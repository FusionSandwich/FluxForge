import csv
import json
from pathlib import Path


def main() -> None:
    output_dir = Path("FluxForge/artifacts/validation/pra_spectrum")
    ref_path = output_dir / "reference_peaks.json"
    ff_path = output_dir / "fluxforge_peaks.json"

    if not (ref_path.exists() and ff_path.exists()):
        return

    ref = json.loads(ref_path.read_text(encoding="utf-8"))["peaks"]
    ff = json.loads(ff_path.read_text(encoding="utf-8"))["peaks"]

    ref_energies = [row[3] for row in ref if row[3] is not None]
    ff_energies = [row[3] for row in ff if row[3] is not None]

    comparisons = []
    for energy in ref_energies:
        closest = (
            min(ff_energies, key=lambda e: abs(e - energy)) if ff_energies else None
        )
        if closest is None:
            continue
        diff_pct = 100.0 * abs(closest - energy) / energy if energy else 0.0
        comparisons.append([energy, closest, diff_pct])

    with (output_dir / "pra_peak_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["ref_energy_keV", "fluxforge_energy_keV", "pct_diff"])
        writer.writerows(comparisons)


if __name__ == "__main__":
    main()
