import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fluxforge.cli.app import build_parser
from fluxforge_gui.parity_registry import GUI_PARITY_REGISTRY


def _cli_commands() -> set[str]:
    parser = build_parser()
    subparser_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return set(subparser_action.choices)


def test_gui_parity_registry_covers_all_cli_commands():
    cli_commands = _cli_commands()
    registry_commands = {record.command for record in GUI_PARITY_REGISTRY}

    assert registry_commands == cli_commands


def test_only_gui_launcher_is_exempt_from_parity():
    exempt_records = [record for record in GUI_PARITY_REGISTRY if record.exempt_reason]

    assert len(exempt_records) == 1
    assert exempt_records[0].command == "gui"

    for record in GUI_PARITY_REGISTRY:
        if record.command == "gui":
            continue
        assert record.gui_surface
        assert record.acceptance_test
