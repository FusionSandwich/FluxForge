from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"
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


def test_no_external_repo_path_literals_in_tests() -> None:
    offenders: list[str] = []

    for path in sorted(TESTS_ROOT.glob("*.py")):
        if path.name in EXCLUDED_FILES:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "np.testing" in line:
                continue

            if any(pattern.search(line) for pattern in BANNED_PATTERNS):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {stripped}")

    assert (
        not offenders
    ), "Found external testing-repo path literals in FluxForge tests:\n" + "\n".join(
        offenders
    )
