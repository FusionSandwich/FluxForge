"""Production-copy and GUI action-catalog contracts.

This module deliberately has no Qt dependency.  It can therefore be used by
the native GUI, command-line validation, packaging checks, and test tooling
without constructing a :class:`QApplication`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence


ACTION_CATALOG_SCHEMA_VERSION = 1


class CopyScope(str, Enum):
    """Visibility scope for user-facing copy."""

    PRODUCTION = "production"
    DEVELOPER = "developer"
    ABOUT_DIAGNOSTICS = "about-diagnostics"


class ActionKind(str, Enum):
    """Supported kinds of interactive GUI surfaces."""

    ACTION = "action"
    BUTTON = "button"
    TAB = "tab"
    EDITOR = "editor"
    CONTEXT_MENU = "context-menu"
    PLOT_TOOL = "plot-tool"


@dataclass(frozen=True)
class ProductionCopy:
    """One piece of visible copy and the surface on which it appears."""

    text: str
    source: str = ""
    scope: CopyScope = CopyScope.PRODUCTION


@dataclass(frozen=True)
class CopyViolation:
    """A forbidden production-copy match."""

    source: str
    token: str
    text: str

    def describe(self) -> str:
        location = self.source or "<unknown surface>"
        return f"{location}: forbidden production copy {self.token!r} in {self.text!r}"


@dataclass(frozen=True)
class ActionCatalogEntry:
    """Stable metadata for an interactive GUI action.

    ``action_id`` is a durable automation and telemetry identifier.  It must
    describe intent rather than presentation so labels and layouts can change
    without invalidating tests.
    """

    action_id: str
    kind: ActionKind
    label: str
    surface: str
    test_ids: tuple[str, ...]
    scope: CopyScope = CopyScope.PRODUCTION
    description: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ActionCatalogEntry":
        """Create an entry from JSON-compatible catalog data."""

        raw_tests = value.get("test_ids", ())
        if isinstance(raw_tests, str):
            test_ids = (raw_tests,)
        elif isinstance(raw_tests, Sequence):
            test_ids = tuple(str(item) for item in raw_tests)
        else:
            test_ids = ()
        return cls(
            action_id=str(value.get("action_id", "")),
            kind=ActionKind(str(value.get("kind", ""))),
            label=str(value.get("label", "")),
            surface=str(value.get("surface", "")),
            test_ids=test_ids,
            scope=CopyScope(str(value.get("scope", CopyScope.PRODUCTION.value))),
            description=str(value.get("description", "")),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "label": self.label,
            "surface": self.surface,
            "test_ids": list(self.test_ids),
            "scope": self.scope.value,
            "description": self.description,
        }


@dataclass(frozen=True)
class CatalogViolation:
    """A validation failure in an action catalog."""

    action_id: str
    field: str
    message: str

    def describe(self) -> str:
        return f"{self.action_id or '<missing action id>'}.{self.field}: {self.message}"


class ProductionContractError(ValueError):
    """Raised when production copy or an action catalog is invalid."""


@dataclass(frozen=True)
class ActionCatalog:
    """Versioned collection of interactive GUI action metadata."""

    entries: tuple[ActionCatalogEntry, ...]
    schema_version: int = ACTION_CATALOG_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ActionCatalog":
        """Create a catalog from a JSON-compatible mapping."""

        raw_entries = value.get("entries", ())
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, str):
            raise ValueError("action catalog 'entries' must be a sequence")
        entries: list[ActionCatalogEntry] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise ValueError("each action catalog entry must be a mapping")
            entries.append(ActionCatalogEntry.from_mapping(raw_entry))
        return cls(
            entries=tuple(entries),
            schema_version=int(value.get("schema_version", 0)),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "entries": [entry.as_dict() for entry in self.entries],
        }


# These patterns describe implementation status rather than an analyst task.
# Scientific uses of words such as "phase" remain valid; only numbered roadmap
# phases are rejected.  Developer and About/Diagnostics surfaces are explicitly
# exempt because their purpose is to expose implementation metadata.
FORBIDDEN_PRODUCTION_COPY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("numbered phase", re.compile(r"\bphase\s+\d+(?:\.\d+)*\b", re.IGNORECASE)),
    ("roadmap", re.compile(r"\broadmap\b", re.IGNORECASE)),
    ("planned", re.compile(r"\bplanned\b", re.IGNORECASE)),
    ("placeholder", re.compile(r"\bplaceholder\b", re.IGNORECASE)),
    ("coming soon", re.compile(r"\bcoming\s+soon\b", re.IGNORECASE)),
    (
        "future capability",
        re.compile(r"\bfuture\s+capabilit(?:y|ies)\b", re.IGNORECASE),
    ),
    ("reserved for", re.compile(r"\breserved\s+for\b", re.IGNORECASE)),
    ("Qt stack", re.compile(r"\bqt\s+stack\b", re.IGNORECASE)),
    ("legacy GUI", re.compile(r"\blegacy\s+gui\b", re.IGNORECASE)),
    ("parity target", re.compile(r"\bparity\s+target\b", re.IGNORECASE)),
    ("design target", re.compile(r"\bdesign\s+target\b", re.IGNORECASE)),
    ("bGamma", re.compile(r"\bbgamma(?:-style)?\b", re.IGNORECASE)),
)

DEVELOPER_COPY_SCOPES = frozenset({CopyScope.DEVELOPER, CopyScope.ABOUT_DIAGNOSTICS})

_ACTION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


def lint_production_copy(
    items: Iterable[str | ProductionCopy],
) -> tuple[CopyViolation, ...]:
    """Return forbidden narration found on production-visible surfaces.

    Plain strings are treated as production copy.  Callers must opt into an
    exemption by supplying :class:`ProductionCopy` with a developer or
    About/Diagnostics scope; filename conventions never create exemptions.
    """

    violations: list[CopyViolation] = []
    for raw_item in items:
        item = (
            raw_item
            if isinstance(raw_item, ProductionCopy)
            else ProductionCopy(raw_item)
        )
        if item.scope in DEVELOPER_COPY_SCOPES:
            continue
        for token, pattern in FORBIDDEN_PRODUCTION_COPY:
            if pattern.search(item.text):
                violations.append(
                    CopyViolation(source=item.source, token=token, text=item.text)
                )
    return tuple(violations)


def assert_production_copy(items: Iterable[str | ProductionCopy]) -> None:
    """Raise when any production-visible copy contains forbidden narration."""

    violations = lint_production_copy(items)
    if violations:
        details = "\n".join(violation.describe() for violation in violations)
        raise ProductionContractError(f"Production copy validation failed:\n{details}")


def validate_action_catalog(
    entries: Iterable[ActionCatalogEntry] | ActionCatalog,
    *,
    require_test_coverage: bool = True,
) -> tuple[CatalogViolation, ...]:
    """Validate stable IDs, visible copy, and behavioral-test coverage."""

    violations: list[CatalogViolation] = []
    if isinstance(entries, ActionCatalog):
        catalog = entries.entries
        if entries.schema_version != ACTION_CATALOG_SCHEMA_VERSION:
            violations.append(
                CatalogViolation(
                    "<catalog>",
                    "schema_version",
                    f"expected {ACTION_CATALOG_SCHEMA_VERSION}, got "
                    f"{entries.schema_version}",
                )
            )
    else:
        catalog = tuple(entries)
    id_counts: dict[str, int] = {}
    for entry in catalog:
        id_counts[entry.action_id] = id_counts.get(entry.action_id, 0) + 1

    for entry in catalog:
        action_id = entry.action_id
        if not _ACTION_ID_PATTERN.fullmatch(action_id):
            violations.append(
                CatalogViolation(
                    action_id,
                    "action_id",
                    "must be a lower-case dotted identifier such as "
                    "'spectrum.roi.create'",
                )
            )
        if action_id and id_counts[action_id] > 1:
            violations.append(
                CatalogViolation(action_id, "action_id", "must be unique")
            )
        if not isinstance(entry.kind, ActionKind):
            violations.append(
                CatalogViolation(action_id, "kind", "must be an ActionKind")
            )
        if not isinstance(entry.scope, CopyScope):
            violations.append(
                CatalogViolation(action_id, "scope", "must be a CopyScope")
            )
        if not entry.label.strip():
            violations.append(CatalogViolation(action_id, "label", "must not be blank"))
        if not entry.surface.strip():
            violations.append(
                CatalogViolation(action_id, "surface", "must not be blank")
            )
        has_test_coverage = any(test_id.strip() for test_id in entry.test_ids)
        if require_test_coverage and not has_test_coverage:
            violations.append(
                CatalogViolation(
                    action_id,
                    "test_ids",
                    "must reference at least one behavioral test",
                )
            )
        if any(not test_id.strip() for test_id in entry.test_ids):
            violations.append(
                CatalogViolation(action_id, "test_ids", "must not contain blank IDs")
            )

        copy_violations = lint_production_copy(
            (
                ProductionCopy(
                    text=entry.label,
                    source=f"{action_id}.label",
                    scope=entry.scope,
                ),
                ProductionCopy(
                    text=entry.description,
                    source=f"{action_id}.description",
                    scope=entry.scope,
                ),
            )
        )
        violations.extend(
            CatalogViolation(action_id, "copy", violation.describe())
            for violation in copy_violations
        )

    return tuple(violations)


def assert_valid_action_catalog(
    entries: Iterable[ActionCatalogEntry] | ActionCatalog,
    *,
    require_test_coverage: bool = True,
) -> None:
    """Raise when an action catalog violates the production contract."""

    violations = validate_action_catalog(
        entries,
        require_test_coverage=require_test_coverage,
    )
    if violations:
        details = "\n".join(violation.describe() for violation in violations)
        raise ProductionContractError(f"Action catalog validation failed:\n{details}")


__all__ = [
    "ACTION_CATALOG_SCHEMA_VERSION",
    "DEVELOPER_COPY_SCOPES",
    "FORBIDDEN_PRODUCTION_COPY",
    "ActionCatalog",
    "ActionCatalogEntry",
    "ActionKind",
    "CatalogViolation",
    "CopyScope",
    "CopyViolation",
    "ProductionContractError",
    "ProductionCopy",
    "assert_production_copy",
    "assert_valid_action_catalog",
    "lint_production_copy",
    "validate_action_catalog",
]
