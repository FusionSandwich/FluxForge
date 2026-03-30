"""Jinja2-backed reporting engine for the modern Qt shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _load_weasyprint_html():
    """Load WeasyPrint's HTML entrypoint on demand."""

    try:
        module = import_module("weasyprint")
    except ImportError as exc:  # pragma: no cover - optional dependency branch
        raise RuntimeError(
            "PDF export requires the optional reporting extra (`WeasyPrint`)."
        ) from exc
    return module.HTML


@dataclass(frozen=True)
class ReportTemplateSpec:
    """Minimal report-template description."""

    name: str
    description: str
    required_context_keys: tuple[str, ...] = ()
    template_file: str | None = None


@dataclass(frozen=True)
class ReportRenderResult:
    """Rendered report payload."""

    template_name: str
    html: str
    context: dict[str, Any]


@dataclass
class ReportingEngine:
    """Jinja2 template registry and export surface."""

    templates: dict[str, ReportTemplateSpec] = field(default_factory=dict)
    template_dir: Path = field(default_factory=lambda: DEFAULT_TEMPLATE_DIR)

    def __post_init__(self) -> None:
        self.template_dir = Path(self.template_dir)
        self._environment = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        if not self.templates:
            self._register_bundled_templates()

    def register(self, template: ReportTemplateSpec) -> None:
        self.templates[template.name] = template

    def _register_bundled_templates(self) -> None:
        self.register(
            ReportTemplateSpec(
                name="standard_lab",
                description="Standard laboratory review template with residuals and provenance.",
                required_context_keys=(
                    "title",
                    "spectrum_image",
                    "calibration_curve",
                    "calibration_residuals",
                    "efficiency_curve",
                    "efficiency_residuals",
                    "residuals_grid",
                    "peak_table",
                    "activity_table",
                    "astm_status_table",
                    "qa_status_snapshot",
                    "provenance",
                ),
                template_file="standard_lab.html.j2",
            )
        )
        self.register(
            ReportTemplateSpec(
                name="astm_compliance",
                description="ASTM-oriented compliance report with QA and lock summaries.",
                required_context_keys=(
                    "title",
                    "astm_status_table",
                    "qa_status_snapshot",
                    "peak_table",
                    "activity_table",
                    "residuals_grid",
                    "provenance",
                ),
                template_file="astm_compliance.html.j2",
            )
        )
        self.register(
            ReportTemplateSpec(
                name="batch_summary",
                description="Aggregate batch-analysis summary with per-spectrum rows.",
                required_context_keys=("title", "batch_rows", "aggregate_csv", "provenance"),
                template_file="batch_summary.html.j2",
            )
        )

    def bundled_templates(self) -> tuple[ReportTemplateSpec, ...]:
        return tuple(self.templates.values())

    def can_export_pdf(self) -> bool:
        """Return whether PDF export support is available in the environment."""

        try:
            _load_weasyprint_html()
        except RuntimeError:
            return False
        return True

    def render(self, template_name: str, context: dict[str, Any]) -> ReportRenderResult:
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
        template_file = template.template_file or f"{template_name}.html.j2"
        html = self._environment.get_template(template_file).render(**context)
        return ReportRenderResult(template_name=template_name, html=html, context=dict(context))

    def export_html(
        self,
        template_name: str,
        context: dict[str, Any],
        path: str | Path,
    ) -> Path:
        rendered = self.render(template_name, context)
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(rendered.html, encoding="utf-8")
        return resolved

    def export_pdf(
        self,
        template_name: str,
        context: dict[str, Any],
        path: str | Path,
    ) -> Path:
        """Render one bundled template to PDF via WeasyPrint."""

        rendered = self.render(template_name, context)
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        html_class = _load_weasyprint_html()
        html_class(string=rendered.html, base_url=str(self.template_dir)).write_pdf(
            str(resolved)
        )
        return resolved
