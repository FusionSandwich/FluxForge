#!/usr/bin/env python3
"""
Cross-section library demo using placeholder IRDFF data.
"""

from fluxforge.data.crosssections import create_irdff_placeholder_library


def main() -> None:
    library = create_irdff_placeholder_library()
    matches = library.search(target="Au-197", outgoing="g")

    if not matches:
        print("No Au-197 capture data found.")
        return

    xs = matches[0]
    value = xs.evaluate(1e-6)
    print(f"{xs.reaction}: σ(1e-6 MeV) = {value:.3e} b")


if __name__ == "__main__":
    main()
