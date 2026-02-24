#!/usr/bin/env python3
"""
IEC spectrum reader demo.

Usage:
    python iec_read_demo.py /path/to/file.iec
"""

import sys
from pathlib import Path

from fluxforge.io.iec import read_iec_file


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python iec_read_demo.py /path/to/file.iec")
        return 1

    path = Path(sys.argv[1])
    spectrum = read_iec_file(path)

    print(f"IEC file: {path.name}")
    print(f"Channels: {spectrum.counts.size}")
    print(f"Live time: {spectrum.live_time:.2f} s")
    print(f"Real time: {spectrum.real_time:.2f} s")
    print(f"Detector ID: {spectrum.detector_id}")
    print(f"Energy cal: {spectrum.calibration.get('energy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
