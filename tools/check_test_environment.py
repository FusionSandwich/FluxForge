"""Import smoke check for the complete environment declared in environment.yml."""

from __future__ import annotations

import importlib
from importlib import metadata
import json
from pathlib import Path
import sys


MODULES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "h5py": "h5py",
    "lxml.etree": "lxml",
    "PySide6.QtWidgets": "PySide6",
    "pyqtgraph": "pyqtgraph",
    "tensorflow": "tensorflow",
    "jinja2": "Jinja2",
    "weasyprint": "WeasyPrint",
    "PyInstaller": "pyinstaller",
    "pyunfold": "pyunfold",
    "pytest": "pytest",
    "black": "black",
    "flake8": "flake8",
}


def main() -> int:
    results = {}
    errors = []
    if sys.version_info[:2] != (3, 11):
        errors.append("The complete test environment requires Python 3.11")
    for module_name, distribution in MODULES.items():
        try:
            module = importlib.import_module(module_name)
            version = metadata.version(distribution) if distribution else "stdlib"
            results[module_name] = {"version": version, "path": module.__file__}
            if module_name == "tensorflow" and version != "2.15.1":
                errors.append(f"Expected TensorFlow 2.15.1, found {version}")
            if module_name == "numpy" and not version.startswith("1.26."):
                errors.append(f"Expected NumPy 1.26.x, found {version}")
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    try:
        import numpy as np
        import tensorflow as tf
        from weasyprint import HTML

        product = tf.matmul(tf.constant([[2.0]]), tf.constant([[3.0]])).numpy()
        if not np.array_equal(product, [[6.0]]):
            errors.append("TensorFlow CPU multiplication returned an unexpected value")
        if (
            not HTML(string="<p>FluxForge environment smoke check</p>")
            .write_pdf()
            .startswith(b"%PDF-")
        ):
            errors.append("WeasyPrint did not create a PDF")
    except Exception as exc:
        errors.append(f"Runtime smoke: {type(exc).__name__}: {exc}")
    report = {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "imports": results,
        "errors": errors,
        "passed": not errors,
    }
    print(json.dumps(report, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
