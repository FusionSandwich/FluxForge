"""Run all collected tests, isolating GUI batches and retaining case evidence.

Usage: python tools/run_acceptance.py --output artifacts/validation/local_acceptance
Optional dependencies and reference fixtures must be installed/bound beforehand.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class CaseRecorder:
    def __init__(self, output, name, group):
        self.output, self.name, self.group = output, name, group

    def pytest_collection_modifyitems(self, config, items):
        groups, cache, selected, deselected = {}, {}, [], []
        for item in items:
            path = Path(item.path)
            if path not in cache:
                source = path.read_text(encoding="utf-8")
                cache[path] = any(
                    token in source
                    for token in (
                        "QT_AVAILABLE",
                        "QApplication",
                        "QTest",
                        "PYQTGRAPH_AVAILABLE",
                    )
                ) or any(token in path.name for token in ("gui", "pyqtgraph"))
            group = "gui" if cache[path] else "core"
            groups[item.nodeid] = group
            target = selected if self.group in (group, "all") else deselected
            target.append(item)
        write_json(self.output / f"{self.name}.selection.json", groups)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.outcome != "passed":
            with (self.output / f"{self.name}.cases.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(
                    json.dumps(
                        {
                            "nodeid": report.nodeid,
                            "phase": report.when,
                            "outcome": report.outcome,
                            "duration_s": report.duration,
                            "detail": (
                                str(report.longrepr)
                                if report.outcome != "passed"
                                else ""
                            ),
                        }
                    )
                    + "\n"
                )


def worker():
    import pytest

    output, name, group = Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    return pytest.main(sys.argv[5:], plugins=[CaseRecorder(output, name, group)])


def stop_process(process):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def run_batch(output, name, group, args, timeout, environment):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        str(output),
        name,
        group,
        *args,
    ]
    started = time.monotonic()
    timed_out = False
    with (output / f"{name}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process(process)
            timed_out, code = True, 124
        except KeyboardInterrupt:
            stop_process(process)
            raise
    receipt = {
        "name": name,
        "argv": command,
        "exit_code": code,
        "timed_out": timed_out,
        "elapsed_s": time.monotonic() - started,
    }
    write_json(output / f"{name}.run.json", receipt)
    print(f"{name}: exit {code}, {receipt['elapsed_s']:.1f}s", flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tests", nargs="+", default=["tests"])
    parser.add_argument("--gui-batch-size", type=int, default=5)
    parser.add_argument("--core-timeout", type=int, default=1200)
    parser.add_argument("--gui-timeout", type=int, default=120)
    parser.add_argument("--require-no-skips", action="store_true")
    args = parser.parse_args()
    if min(args.gui_batch_size, args.core_timeout, args.gui_timeout) <= 0:
        parser.error("Batch size and timeouts must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory to preserve earlier evidence")
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "QT_QPA_PLATFORM": "offscreen",
            "MPLBACKEND": "Agg",
            "PYTHONIOENCODING": "utf-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
        }
    )
    source_paths = sorted(
        p
        for folder in (ROOT / "src", ROOT / "tests")
        for p in folder.rglob("*.py")
        if "__pycache__" not in p.parts
    )
    digest = hashlib.sha256()
    for path in source_paths + [ROOT / "pyproject.toml"]:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT)
    write_json(
        output / "environment.json",
        {
            "checkout": str(ROOT),
            "commit": git("rev-parse", "HEAD").decode().strip(),
            "source_sha256": digest.hexdigest(),
            "diff_sha256": hashlib.sha256(git("diff", "HEAD")).hexdigest(),
            "python": sys.version,
            "dependencies": {
                dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions()
            },
            "fixture_bindings": {
                key: environment.get(key)
                for key in (
                    "FLUXFORGE_REFERENCE_ROOT",
                    "FLUXFORGE_TRANSPORT_FIXTURE_DIR",
                )
            },
        },
    )
    collected = run_batch(
        output,
        "collection",
        "all",
        ["--collect-only", "-q", *args.tests],
        args.core_timeout,
        environment,
    )
    if collected["exit_code"]:
        return collected["exit_code"]
    selection = json.loads((output / "collection.selection.json").read_text())
    plan = []
    if "core" in selection.values():
        plan.append(("core", "core", args.tests, args.core_timeout))
    gui = defaultdict(list)
    for node, group in selection.items():
        if group == "gui":
            gui[node.split("::")[0]].append(node)
    for nodes in gui.values():
        for start in range(0, len(nodes), args.gui_batch_size):
            plan.append(
                (
                    f"gui_{len(plan)}",
                    "gui",
                    nodes[start : start + args.gui_batch_size],
                    args.gui_timeout,
                )
            )
    receipts, outcomes = [], {}
    priority = {"passed": 0, "skipped": 1, "failed": 2}
    for name, group, nodes, timeout in plan:
        receipt = run_batch(
            output,
            name,
            group,
            ["-q", "-ra", f"--junitxml={output / (name + '.xml')}", *nodes],
            timeout,
            environment,
        )
        receipts.append(receipt)
        path = output / f"{name}.cases.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                previous = outcomes.get(record["nodeid"])
                if (
                    previous is None
                    or priority[record["outcome"]] >= priority[previous["outcome"]]
                ):
                    outcomes[record["nodeid"]] = record
    counts = dict(Counter(record["outcome"] for record in outcomes.values()))
    missing, unexpected = sorted(set(selection) - outcomes.keys()), sorted(
        outcomes.keys() - set(selection)
    )
    passed = not (
        missing
        or unexpected
        or counts.get("failed")
        or any(r["exit_code"] for r in receipts)
        or (args.require_no_skips and counts.get("skipped"))
    )
    write_json(output / "case_outcomes.json", outcomes)
    write_json(
        output / "summary.json",
        {
            "passed": passed,
            "collected": len(selection),
            "counts": counts,
            "missing": missing,
            "unexpected": unexpected,
            "runs": receipts,
        },
    )
    print(json.dumps({"passed": passed, "collected": len(selection), "counts": counts}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(
        worker() if len(sys.argv) > 1 and sys.argv[1] == "--worker" else main()
    )
