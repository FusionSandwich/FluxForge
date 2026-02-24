import json
import runpy
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


def main() -> None:
    output_dir = Path("FluxForge/artifacts/validation/pyfindpeaks")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = runpy.run_path("testing/py-findpeaks/tests/vector.py")
    vector = np.array(data["vector"], dtype=float)
    indices, _ = find_peaks(vector)

    (output_dir / "pyfindpeaks_reference.json").write_text(
        json.dumps({"peaks": indices.tolist()}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
