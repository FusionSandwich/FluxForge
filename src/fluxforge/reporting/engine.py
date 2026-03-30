"""Reporting scaffolding for the next-generation report engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReportTemplateSpec:
    """Minimal report-template description."""

    name: str
    description: str
    required_context_keys: tuple[str, ...] = ()


@dataclass
class ReportingEngine:
    """Placeholder reporting engine registry surface."""

    templates: dict[str, ReportTemplateSpec] = field(default_factory=dict)

    def register(self, template: ReportTemplateSpec) -> None:
        self.templates[template.name] = template

    def render(self, template_name: str, context: dict[str, Any]) -> dict[str, Any]:
        if template_name not in self.templates:
            raise KeyError(f"Unknown template {template_name!r}")
        template = self.templates[template_name]
        missing = [
            key for key in template.required_context_keys if key not in context
        ]
        if missing:
            raise KeyError(
                f"Template {template_name!r} missing context keys: {', '.join(missing)}"
            )
        return {"template": template_name, "context": context}
