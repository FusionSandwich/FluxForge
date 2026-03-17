from __future__ import annotations

import re
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
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if skip_comments and stripped.startswith("#"):
                continue
            if "np.testing" in line:
                continue

            if any(pattern.search(line) for pattern in BANNED_PATTERNS):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {stripped}")

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
