from __future__ import annotations

import re
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"
SRC_ROOT = REPO_ROOT / "src"
SELF_NAME = Path(__file__).name
EXCLUDED_FILES = {
    "audit_capabilities.py",  # Contains descriptive references, not runtime data paths.
    SELF_NAME,
}

# Runtime path literals that would make tests depend on sibling ALARA repos.
BANNED_PATTERNS = (
    re.compile(r"ALARA/testing"),
    re.compile(r"/filespace/.*/ALARA/testing"),
    re.compile(r"/groupspace/.*/ALARA/testing"),
    re.compile(r"""Path\(\s*["'](?:\.\./)*testing/"""),
    re.compile(r"""["'](?:\.\./)*testing/"""),
)

TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".md"}


def _find_offenders(root: Path, *, skip_comments: bool) -> list[str]:
    offenders: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if root == TESTS_ROOT and path.name in EXCLUDED_FILES:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.name == "manifest.json":
            payload = json.loads(text)
            if (
                isinstance(payload, dict)
                and payload.get("source_repo")
                and payload.get("fixture_id")
            ):
                # Historical source locations are provenance, not runtime paths.
                # Keep scanning every input/output locator in the same manifest.
                payload.pop("source_paths", None)
                text = json.dumps(payload, indent=2)
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if skip_comments and stripped.startswith("#"):
                continue
            if "np.testing" in line:
                continue

            if any(pattern.search(line) for pattern in BANNED_PATTERNS):
                display_path = (
                    path.relative_to(REPO_ROOT)
                    if path.is_relative_to(REPO_ROOT)
                    else path.relative_to(root)
                )
                offenders.append(f"{display_path}:{lineno}: {stripped}")

    return offenders


def test_no_external_repo_path_literals_in_tests() -> None:
    offenders = _find_offenders(TESTS_ROOT, skip_comments=True)

    assert (
        not offenders
    ), "Found external testing-repo path literals in FluxForge tests:\n" + "\n".join(
        offenders
    )


def test_no_external_repo_path_literals_in_src() -> None:
    offenders = _find_offenders(SRC_ROOT, skip_comments=False)

    assert (
        not offenders
    ), "Found external testing-repo path literals in FluxForge src:\n" + "\n".join(
        offenders
    )


def test_provenance_exception_does_not_hide_runtime_inputs(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "fixture_id": "contract",
                "source_repo": "upstream/repo",
                "source_paths": ["testing/upstream/historical.dat"],
                "input_files": ["local.dat"],
            }
        )
    )
    assert _find_offenders(tmp_path, skip_comments=True) == []
    payload = json.loads(manifest.read_text())
    payload["input_files"] = ["testing/upstream/runtime.dat"]
    manifest.write_text(json.dumps(payload))
    offenders = _find_offenders(tmp_path, skip_comments=True)
    assert len(offenders) == 1 and "runtime.dat" in offenders[0]
