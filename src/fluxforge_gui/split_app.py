"""Developer utility for splitting `fluxforge_gui.app` into smaller modules."""

import ast
import sys
from pathlib import Path


def get_method_lines(source, class_name):
    tree = ast.parse(source)
    methods = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    methods[child.name] = (child.lineno, child.end_lineno)
    return methods


def resolve_app_path(argv: list[str]) -> Path:
    """Return the app module path to split, defaulting to the local package file."""

    if len(argv) > 1:
        return Path(argv[1]).expanduser().resolve()
    return Path(__file__).resolve().with_name("app.py")


def main():
    app_path = resolve_app_path(sys.argv)
    source = app_path.read_text(encoding="utf-8")
    lines = source.split("\n")

    methods = get_method_lines(source, "FluxForgeGui")

    ui_builder_methods = set(
        [
            name
            for name in methods
            if name.startswith("_build_")
            or name == "_configure_styles"
            or name.endswith("_row")
        ]
    )

    command_methods = set(
        [
            name
            for name in methods
            if name.startswith("_run_") or name.startswith("_curie_run_")
        ]
    )

    ui_builder_lines = []
    command_lines = []

    remove_lines = set()

    for m in sorted(ui_builder_methods, key=lambda x: methods[x][0]):
        start, end = methods[m]
        ui_builder_lines.extend(lines[start - 1 : end])
        for i in range(start - 1, end):
            remove_lines.add(i)
        ui_builder_lines.append("")

    for m in sorted(command_methods, key=lambda x: methods[x][0]):
        start, end = methods[m]
        command_lines.extend(lines[start - 1 : end])
        for i in range(start - 1, end):
            remove_lines.add(i)
        command_lines.append("")

    app_lines = []
    for i, line in enumerate(lines):
        if i not in remove_lines:
            app_lines.append(line)

    app_path.write_text("\n".join(app_lines), encoding="utf-8")

    ui_path = app_path.parent / "ui_builder.py"
    ui_path.write_text(
        "import tkinter as tk\nfrom tkinter import ttk\n\nclass UiBuilderMixin:\n"
        + "\n".join(ui_builder_lines),
        encoding="utf-8",
    )

    cmd_path = app_path.parent / "commands.py"
    cmd_path.write_text(
        (
            "import tkinter as tk\n"
            "from tkinter import messagebox\n"
            "from pathlib import Path\n"
            "from argparse import Namespace\n"
            "from fluxforge import cli_app\n\n"
            "class CommandsMixin:\n"
        )
        + "\n".join(command_lines),
        encoding="utf-8",
    )
    print(f"app.py is now {len(app_lines)} lines.")
    print(f"Created ui_builder.py ({len(ui_builder_lines)} lines)")
    print(f"Created commands.py ({len(command_lines)} lines)")


if __name__ == "__main__":
    main()
