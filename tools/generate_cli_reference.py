#!/usr/bin/env python3
"""Generate the committed FluxForge CLI reference Markdown file."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluxforge.cli.app import build_parser
from fluxforge.cli.command_catalog import (
    build_command_catalog,
    render_command_catalog_markdown,
)


def main() -> int:
    parser = build_parser()
    catalog = build_command_catalog(parser)
    output_path = ROOT / "docs" / "CLI_REFERENCE.md"
    output_path.write_text(
        render_command_catalog_markdown(catalog),
        encoding="utf-8",
    )
    print(f"Wrote CLI reference to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
