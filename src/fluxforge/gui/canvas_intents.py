"""Renderer-independent interaction events for scientific canvases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping


CANVAS_INTENT_SCHEMA = "fluxforge.canvas_intent.v1"
CANVAS_INTENT_VERSION = 1

_INTENT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "kind",
        "spectrum_id",
        "roi_id",
        "peak_id",
        "component_ids",
        "bounds",
        "left_background",
        "right_background",
        "position",
        "nuclide",
        "tag",
        "role",
        "viewport_id",
        "x_range",
        "y_range",
        "enabled",
        "drag_token",
    }
)


class CanvasIntentKind(str, Enum):
    """Stable operations emitted by a spectrum renderer."""

    CREATE_ROI = "roi.create"
    SELECT_ROI = "roi.select"
    DELETE_ROI = "roi.delete"
    MOVE_ROI = "roi.move"
    MOVE_BACKGROUND = "roi.background.move"
    ADD_PEAK = "peak.add"
    MOVE_PEAK = "peak.move"
    DELETE_PEAK = "peak.delete"
    SPLIT_PEAK = "peak.split"
    MERGE_PEAKS = "peak.merge"
    ASSIGN_NUCLIDE = "nuclide.assign"
    CLEAR_NUCLIDE = "nuclide.clear"
    PIN_NUCLIDE = "nuclide.pin"
    UNPIN_NUCLIDE = "nuclide.unpin"
    TAG_NUCLIDE = "nuclide.tag"
    ASSIGN_SPECTRUM_ROLE = "spectrum.role.assign"
    CHANGE_VIEWPORT = "viewport.change"
    TOGGLE_CROSSHAIR = "viewport.crosshair"
    SET_LOG_SCALE = "viewport.log_scale"
    SET_LABELS_VISIBLE = "viewport.labels"


_ROI_ID_KINDS = {
    CanvasIntentKind.SELECT_ROI,
    CanvasIntentKind.DELETE_ROI,
    CanvasIntentKind.MOVE_ROI,
    CanvasIntentKind.MOVE_BACKGROUND,
}
_PEAK_ID_KINDS = {
    CanvasIntentKind.MOVE_PEAK,
    CanvasIntentKind.DELETE_PEAK,
    CanvasIntentKind.SPLIT_PEAK,
    CanvasIntentKind.ASSIGN_NUCLIDE,
    CanvasIntentKind.CLEAR_NUCLIDE,
}
_BOUNDS_KINDS = {
    CanvasIntentKind.CREATE_ROI,
    CanvasIntentKind.MOVE_ROI,
}
_ENABLED_KINDS = {
    CanvasIntentKind.TOGGLE_CROSSHAIR,
    CanvasIntentKind.SET_LOG_SCALE,
    CanvasIntentKind.SET_LABELS_VISIBLE,
}


@dataclass(frozen=True)
class CanvasIntent:
    """One validated UI intent independent of any plotting toolkit.

    The event is deliberately descriptive rather than executable.  A controller
    translates it into a focused document edit and, where appropriate, one
    mergeable undo command after a drag completes.
    """

    kind: CanvasIntentKind
    spectrum_id: str | None = None
    roi_id: str | None = None
    peak_id: str | None = None
    component_ids: tuple[str, ...] = ()
    bounds: tuple[float, float] | None = None
    left_background: tuple[float, float] | None = None
    right_background: tuple[float, float] | None = None
    position: float | None = None
    nuclide: str | None = None
    tag: str | None = None
    role: str | None = None
    viewport_id: str | None = None
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    enabled: bool | None = None
    drag_token: str | None = None
    schema: str = CANVAS_INTENT_SCHEMA
    version: int = CANVAS_INTENT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CanvasIntentKind(self.kind))
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` when required intent data is absent or invalid."""

        if self.schema != CANVAS_INTENT_SCHEMA or self.version != CANVAS_INTENT_VERSION:
            raise ValueError("Unsupported CanvasIntent schema or version.")
        if self.kind not in {
            CanvasIntentKind.PIN_NUCLIDE,
            CanvasIntentKind.UNPIN_NUCLIDE,
            CanvasIntentKind.TAG_NUCLIDE,
        } and not _text(self.spectrum_id):
            raise ValueError(f"{self.kind.value} requires spectrum_id.")
        if self.kind in _ROI_ID_KINDS and not _text(self.roi_id):
            raise ValueError(f"{self.kind.value} requires roi_id.")
        if self.kind in _PEAK_ID_KINDS and not _text(self.peak_id):
            raise ValueError(f"{self.kind.value} requires peak_id.")
        if self.kind in _BOUNDS_KINDS:
            _validate_range(self.bounds, "bounds")
        if self.kind is CanvasIntentKind.MOVE_BACKGROUND:
            if self.left_background is None and self.right_background is None:
                raise ValueError("roi.background.move requires a background range.")
            if self.left_background is not None:
                _validate_range(self.left_background, "left_background")
            if self.right_background is not None:
                _validate_range(self.right_background, "right_background")
        if self.kind in {CanvasIntentKind.ADD_PEAK, CanvasIntentKind.MOVE_PEAK}:
            if (
                self.position is None
                or isinstance(self.position, bool)
                or not isfinite(float(self.position))
            ):
                raise ValueError(f"{self.kind.value} requires a finite position.")
        if self.kind is CanvasIntentKind.MERGE_PEAKS:
            if len(set(self.component_ids)) < 2:
                raise ValueError(
                    "peak.merge requires at least two unique component IDs."
                )
        if self.kind is CanvasIntentKind.ASSIGN_NUCLIDE and not _text(self.nuclide):
            raise ValueError("nuclide.assign requires nuclide.")
        if self.kind in {
            CanvasIntentKind.PIN_NUCLIDE,
            CanvasIntentKind.UNPIN_NUCLIDE,
            CanvasIntentKind.TAG_NUCLIDE,
        } and not _text(self.nuclide):
            raise ValueError(f"{self.kind.value} requires nuclide.")
        if self.kind is CanvasIntentKind.TAG_NUCLIDE and not _text(self.tag):
            raise ValueError("nuclide.tag requires tag.")
        if self.kind is CanvasIntentKind.ASSIGN_SPECTRUM_ROLE and not _text(self.role):
            raise ValueError("spectrum.role.assign requires role.")
        if self.kind is CanvasIntentKind.CHANGE_VIEWPORT:
            if not _text(self.viewport_id):
                raise ValueError("viewport.change requires viewport_id.")
            if self.x_range is None and self.y_range is None:
                raise ValueError("viewport.change requires an axis range.")
            if self.x_range is not None:
                _validate_range(self.x_range, "x_range")
            if self.y_range is not None:
                _validate_range(self.y_range, "y_range")
        if self.kind in _ENABLED_KINDS and self.enabled is None:
            raise ValueError(f"{self.kind.value} requires enabled.")

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable representation."""

        payload: dict[str, Any] = {
            "schema": self.schema,
            "version": self.version,
            "kind": self.kind.value,
        }
        for key in (
            "spectrum_id",
            "roi_id",
            "peak_id",
            "position",
            "nuclide",
            "tag",
            "role",
            "viewport_id",
            "enabled",
            "drag_token",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.component_ids:
            payload["component_ids"] = list(self.component_ids)
        for key in (
            "bounds",
            "left_background",
            "right_background",
            "x_range",
            "y_range",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = list(value)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanvasIntent":
        """Parse and validate a serialized intent."""

        if not isinstance(payload, Mapping):
            raise ValueError("CanvasIntent payload must be an object.")
        unknown = sorted(set(payload).difference(_INTENT_FIELDS))
        if unknown:
            raise ValueError(
                "CanvasIntent payload contains unknown fields: " + ", ".join(unknown)
            )
        if payload.get("schema") != CANVAS_INTENT_SCHEMA:
            raise ValueError("Unsupported CanvasIntent schema or version.")
        version = payload.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("CanvasIntent version must be an integer.")

        def pair(name: str) -> tuple[float, float] | None:
            raw = payload.get(name)
            if raw is None:
                return None
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise ValueError(f"{name} must contain two numbers.")
            return (
                _finite_number(raw[0], f"{name}[0]"),
                _finite_number(raw[1], f"{name}[1]"),
            )

        component_ids = payload.get("component_ids", ())
        if not isinstance(component_ids, (list, tuple)):
            raise ValueError("component_ids must be an array.")
        raw_enabled = payload.get("enabled")
        if raw_enabled is not None and not isinstance(raw_enabled, bool):
            raise ValueError("enabled must be a boolean.")
        return cls(
            kind=CanvasIntentKind(str(payload.get("kind") or "")),
            spectrum_id=_optional_text(payload.get("spectrum_id")),
            roi_id=_optional_text(payload.get("roi_id")),
            peak_id=_optional_text(payload.get("peak_id")),
            component_ids=tuple(str(value) for value in component_ids),
            bounds=pair("bounds"),
            left_background=pair("left_background"),
            right_background=pair("right_background"),
            position=(
                _finite_number(payload["position"], "position")
                if payload.get("position") is not None
                else None
            ),
            nuclide=_optional_text(payload.get("nuclide")),
            tag=_optional_text(payload.get("tag")),
            role=_optional_text(payload.get("role")),
            viewport_id=_optional_text(payload.get("viewport_id")),
            x_range=pair("x_range"),
            y_range=pair("y_range"),
            enabled=raw_enabled,
            drag_token=_optional_text(payload.get("drag_token")),
            schema=str(payload.get("schema") or ""),
            version=version,
        )


def _text(value: object) -> str:
    return str(value or "").strip()


def _optional_text(value: object) -> str | None:
    text = _text(value)
    return text or None


def _validate_range(value: tuple[float, float] | None, name: str) -> None:
    if value is None:
        raise ValueError(f"{name} is required.")
    lower = _finite_number(value[0], f"{name}[0]")
    upper = _finite_number(value[1], f"{name}[1]")
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} must be a finite increasing range.")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


__all__ = [
    "CANVAS_INTENT_SCHEMA",
    "CANVAS_INTENT_VERSION",
    "CanvasIntent",
    "CanvasIntentKind",
]
