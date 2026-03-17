import json
from pathlib import Path


def main() -> None:
    output_dir = Path("FluxForge/artifacts/validation/peakfinder_reference")
    ref_path = output_dir / "reference_vector_peaks.json"
    ff_path = output_dir / "fluxforge_vector_peaks.json"

    if not (ref_path.exists() and ff_path.exists()):
        return

    reference = json.loads(ref_path.read_text(encoding="utf-8"))["peaks"]
    fluxforge = json.loads(ff_path.read_text(encoding="utf-8")).get("direct_scipy", [])

    diff = {
        "reference": reference,
        "fluxforge": fluxforge,
        "missing": [idx for idx in reference if idx not in fluxforge],
        "extra": [idx for idx in fluxforge if idx not in reference],
    }

    (output_dir / "peakfinder_comparison.json").write_text(
        json.dumps(diff, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
