"""Phase 6 GUI panels for masking, optimization, and second irradiation."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from fluxforge.analysis.masking_review import (
    rank_line_masking_from_activity_review_payload,
    recommend_alternate_lines,
    summarize_masking_isotopes,
)
from fluxforge.analysis.optimization_bassd import (
    parse_bassd_sweep_payload,
    rank_bassd_schedules,
    serialize_bassd_ranking,
)
from fluxforge.analysis.optimization_difom import (
    parse_difom_sweep_payload,
    rank_difom_schedules,
    serialize_difom_ranking,
)
from fluxforge.analysis.optimization_fim import (
    rank_fim_schedules,
    serialize_fim_ranking,
)
from fluxforge.analysis.optimization_mwdcs import (
    parse_mwdcs_sweep_payload,
    rank_mwdcs_schedules,
    serialize_mwdcs_ranking,
)
from fluxforge.analysis.optimization_schedule_builder import (
    build_difom_payload_from_activity_review,
)
from fluxforge.analysis.optimization_stbdmr import (
    parse_stbdmr_sweep_payload,
    rank_stbdmr_schedules,
    serialize_stbdmr_ranking,
)
from fluxforge.core.activity_review import ActivityReviewResult
from fluxforge.core.analysis_workspace import ActivityCalculationResult
from fluxforge.core.inventory_timeline import (
    DEFAULT_DECAY_SOURCE_ID,
    build_inventory_state_from_payload,
)
from fluxforge.gui.mode_manager import ModeManager
from fluxforge.gui.panels.modern_shell_shared import (
    current_tab_label,
    set_combo_data,
    set_tab_label,
)
from fluxforge.gui.qt_compat import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from fluxforge.io import write_ffexp_bundle
from fluxforge.workflows.irradiation_optimization import (
    build_phase6_support_artifacts,
    build_second_irradiation_candidates,
    plan_second_irradiation,
    serialize_second_irradiation_plan,
)
from fluxforge.workflows.phase6_ldrd_worked_example import (
    DEFAULT_SAMPLE_ID as PHASE6_LDRD_DEFAULT_SAMPLE_ID,
    default_output_root as phase6_ldrd_default_output_root,
    run_phase6_ldrd_worked_example,
)


ActivityReviewProvider = Callable[[], Optional[ActivityReviewResult]]
ActivityResultsProvider = Callable[[], Sequence[ActivityCalculationResult]]


def _parse_csv_floats(raw: str, default: Sequence[float]) -> tuple[float, ...]:
    tokens = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    if not tokens:
        return tuple(float(item) for item in default)
    return tuple(float(item) for item in tokens)


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw or "").split(",") if item.strip())


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if str(key) not in headers:
                headers.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})
    return path


def _activity_review_to_payload(review: ActivityReviewResult | None) -> dict[str, Any] | None:
    if review is None:
        return None
    return review.to_payload()


def _activity_results_to_payload(
    results: Sequence[ActivityCalculationResult],
) -> dict[str, Any] | None:
    rows = list(results)
    if not rows:
        return None
    line_results = []
    isotope_summaries = []
    for index, result in enumerate(rows, start=1):
        line_results.append(
            {
                "peak_id": f"derived-{index}",
                "nuclide": result.nuclide,
                "matched_line_energy_keV": float(result.line_energy_keV),
                "net_counts": max(float(result.age_corrected_activity_bq), 1.0),
                "net_counts_uncertainty": max(
                    float(result.age_corrected_uncertainty_bq or result.uncertainty_bq),
                    1.0,
                ),
            }
        )
        isotope_summaries.append(
            {
                "nuclide": result.nuclide,
                "line_count": 1,
                "total_net_counts": max(float(result.age_corrected_activity_bq), 1.0),
                "half_life_s": float(result.half_life_s),
                "count_time_activity_Bq": float(result.activity_bq),
                "count_time_activity_unc_Bq": float(result.uncertainty_bq),
                "irradiation_time_activity_Bq": float(result.age_corrected_activity_bq),
                "irradiation_time_activity_unc_Bq": float(
                    result.age_corrected_uncertainty_bq or result.uncertainty_bq
                ),
                "dose_rate_uSv_h": max(float(result.age_corrected_activity_bq) * 1.0e-6, 0.0),
            }
        )
    return {
        "schema": "fluxforge.activity_review.v1",
        "source_id": "gui_workspace_proxy",
        "custom_gamma_path": None,
        "live_time_s": 900.0,
        "cooling_time_s": max(
            [float(result.source_age_s or 0.0) for result in rows],
            default=0.0,
        ),
        "line_results": line_results,
        "isotope_summaries": isotope_summaries,
    }


def _current_activity_payload(
    activity_review_provider: ActivityReviewProvider,
    activity_results_provider: ActivityResultsProvider,
) -> dict[str, Any] | None:
    review_payload = _activity_review_to_payload(activity_review_provider())
    if review_payload is not None:
        return review_payload
    return _activity_results_to_payload(activity_results_provider())


def _fill_table(table: QTableWidget, rows: Sequence[Mapping[str, Any]]) -> None:
    row_list = [dict(row) for row in rows]
    headers: list[str] = []
    for row in row_list:
        for key in row.keys():
            if str(key) not in headers:
                headers.append(str(key))
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(row_list))
    for row_index, row in enumerate(row_list):
        for column, header in enumerate(headers):
            item = QTableWidgetItem(str(row.get(header, "")))
            table.setItem(row_index, column, item)


class MaskingReviewPanel(QWidget):
    """Qt surface for line interference and alternate-line guidance."""

    def __init__(
        self,
        *,
        mode_manager: ModeManager,
        activity_review_provider: ActivityReviewProvider,
        activity_results_provider: ActivityResultsProvider,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.mode_manager = mode_manager
        self.activity_review_provider = activity_review_provider
        self.activity_results_provider = activity_results_provider
        self._last_payload: dict[str, Any] | None = None
        self._last_rows: list[dict[str, Any]] = []
        self._last_isotope_rows: list[dict[str, Any]] = []
        self._last_recommendations: list[dict[str, Any]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        intro = QLabel(
            (
                "Line Interference / Masking ranks likely interfering lines, aggregates masking "
                "nuclides, and recommends alternate lines or delayed measurement windows."
            ),
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QGridLayout()
        controls.addWidget(QLabel("Energy window (keV)", self), 0, 0)
        self.energy_window_spin = QDoubleSpinBox(self)
        self.energy_window_spin.setObjectName("MaskingEnergyWindowSpin")
        self.energy_window_spin.setRange(0.1, 20.0)
        self.energy_window_spin.setDecimals(2)
        self.energy_window_spin.setValue(3.0)
        controls.addWidget(self.energy_window_spin, 0, 1)

        controls.addWidget(QLabel("Isotopes of interest", self), 0, 2)
        self.isotopes_filter_edit = QLineEdit(self)
        self.isotopes_filter_edit.setObjectName("MaskingIsotopesFilterEdit")
        self.isotopes_filter_edit.setPlaceholderText("Mo-99,Sc-46")
        controls.addWidget(self.isotopes_filter_edit, 0, 3)

        controls.addWidget(QLabel("Top N", self), 1, 0)
        self.top_n_spin = QSpinBox(self)
        self.top_n_spin.setObjectName("MaskingTopNSpin")
        self.top_n_spin.setRange(1, 500)
        self.top_n_spin.setValue(50)
        controls.addWidget(self.top_n_spin, 1, 1)

        layout.addLayout(controls)

        button_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Masking Review", self)
        self.refresh_button.setObjectName("MaskingRefreshButton")
        self.refresh_button.clicked.connect(self.run_masking_review)
        button_row.addWidget(self.refresh_button)

        self.export_lines_button = QPushButton("Export Line CSV", self)
        self.export_lines_button.setObjectName("MaskingExportLinesButton")
        self.export_lines_button.clicked.connect(self._export_lines_default)
        button_row.addWidget(self.export_lines_button)

        self.export_isotopes_button = QPushButton("Export Isotope CSV", self)
        self.export_isotopes_button.setObjectName("MaskingExportIsotopesButton")
        self.export_isotopes_button.clicked.connect(self._export_isotopes_default)
        button_row.addWidget(self.export_isotopes_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.summary = QLabel("Run an activity review before evaluating masking.", self)
        self.summary.setObjectName("MaskingSummaryLabel")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("MaskingTabs")
        self.line_table = QTableWidget(self)
        self.line_table.setObjectName("MaskingLineTable")
        self.line_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.line_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.line_table, "Line Interference")

        self.isotope_table = QTableWidget(self)
        self.isotope_table.setObjectName("MaskingIsotopeTable")
        self.isotope_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.isotope_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.isotope_table, "Masking Nuclides")

        self.recommendations_browser = QTextBrowser(self)
        self.recommendations_browser.setObjectName("MaskingRecommendationsBrowser")
        self.tabs.addTab(self.recommendations_browser, "Recommendations")
        layout.addWidget(self.tabs, 1)

    def run_masking_review(self) -> dict[str, Any] | None:
        payload = _current_activity_payload(
            self.activity_review_provider,
            self.activity_results_provider,
        )
        if payload is None:
            self.summary.setText("Masking review is unavailable until activity results exist.")
            return None
        selected = _parse_csv_strings(self.isotopes_filter_edit.text())
        ranked = rank_line_masking_from_activity_review_payload(
            payload,
            energy_window_keV=float(self.energy_window_spin.value()),
            isotopes_of_interest=selected,
            top_n=int(self.top_n_spin.value()),
        )
        isotope_rows = summarize_masking_isotopes(
            ranked,
            top_n=int(self.top_n_spin.value()),
        )
        recommendations = [
            item.to_row(index)
            for index, item in enumerate(
                recommend_alternate_lines(
                    payload,
                    ranked,
                    top_n=int(self.top_n_spin.value()),
                ),
                start=1,
            )
        ]
        line_rows = [item.to_row(index) for index, item in enumerate(ranked, start=1)]
        self._last_payload = payload
        self._last_rows = line_rows
        self._last_isotope_rows = isotope_rows
        self._last_recommendations = recommendations
        _fill_table(self.line_table, line_rows)
        _fill_table(self.isotope_table, isotope_rows)
        self.recommendations_browser.setPlainText(
            "\n\n".join(
                [
                    (
                        f"{row['nuclide']}: prefer {float(row['preferred_line_energy_keV']):.3f} keV, "
                        f"guidance={row['guidance']}, masking score={float(row['preferred_masking_score']):.4g}"
                    )
                    for row in recommendations
                ]
            )
            or "No alternate-line recommendations were generated."
        )
        self.summary.setText(
            (
                f"Ranked {len(line_rows)} line interactions across {len(isotope_rows)} masking nuclides. "
                f"Generated {len(recommendations)} alternate-line recommendation(s)."
            )
        )
        return {
            "line_rows": line_rows,
            "isotope_rows": isotope_rows,
            "recommendations": recommendations,
        }

    def export_lines_csv(self, path: str | Path) -> Path:
        if not self._last_rows:
            if self.run_masking_review() is None:
                raise ValueError("Masking review is not available.")
        return _write_csv_rows(Path(path), self._last_rows)

    def export_isotopes_csv(self, path: str | Path) -> Path:
        if not self._last_isotope_rows:
            if self.run_masking_review() is None:
                raise ValueError("Masking review is not available.")
        return _write_csv_rows(Path(path), self._last_isotope_rows)

    def _export_lines_default(self) -> None:
        self.export_lines_csv(Path.cwd() / "masking_lines.csv")

    def _export_isotopes_default(self) -> None:
        self.export_isotopes_csv(Path.cwd() / "masking_isotopes.csv")

    def workflow_state(self) -> dict[str, Any]:
        return {
            "energy_window_keV": float(self.energy_window_spin.value()),
            "isotopes_of_interest": self.isotopes_filter_edit.text(),
            "top_n": int(self.top_n_spin.value()),
            "current_tab": current_tab_label(self.tabs),
        }

    def apply_workflow_state(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        if "energy_window_keV" in payload:
            self.energy_window_spin.setValue(float(payload["energy_window_keV"]))
        if "isotopes_of_interest" in payload:
            self.isotopes_filter_edit.setText(str(payload["isotopes_of_interest"] or ""))
        if "top_n" in payload:
            self.top_n_spin.setValue(max(int(payload["top_n"]), 1))
        if "current_tab" in payload:
            set_tab_label(self.tabs, str(payload["current_tab"]))


class OptimizationWorkspacePanel(QWidget):
    """Qt surface for phase-6 irradiation optimization workflows."""

    def __init__(
        self,
        *,
        mode_manager: ModeManager,
        activity_review_provider: ActivityReviewProvider,
        activity_results_provider: ActivityResultsProvider,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.mode_manager = mode_manager
        self.activity_review_provider = activity_review_provider
        self.activity_results_provider = activity_results_provider
        self._last_activity_payload: dict[str, Any] | None = None
        self._last_output_payload: dict[str, Any] | None = None
        self._last_support_artifacts: dict[str, Any] | None = None
        self._last_worked_example_summary_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        intro = QLabel(
            (
                "Irradiation Optimizer scores schedule candidates with DI-FOM, FIM, MWDCS, "
                "BASS-D, or STBD-MR and surfaces an optimization grid, Pareto-style comparison, "
                "and recommendation card."
            ),
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QGridLayout()
        controls.addWidget(QLabel("Objective", self), 0, 0)
        self.objective_combo = QComboBox(self)
        self.objective_combo.setObjectName("OptimizationObjectiveCombo")
        self.objective_combo.addItem("DI-FOM", "di-fom")
        self.objective_combo.addItem("FIM D-opt", "fim-d")
        self.objective_combo.addItem("FIM A-opt", "fim-a")
        self.objective_combo.addItem("FIM C-opt", "fim-c")
        self.objective_combo.addItem("MWDCS", "mwdcs")
        self.objective_combo.addItem("BASS-D", "bass-d")
        self.objective_combo.addItem("STBD-MR", "stbd-mr")
        controls.addWidget(self.objective_combo, 0, 1)

        controls.addWidget(QLabel("Irradiation grid (s)", self), 0, 2)
        self.irradiation_grid_edit = QLineEdit("1800,3600,7200,14400", self)
        self.irradiation_grid_edit.setObjectName("OptimizationIrradiationGridEdit")
        controls.addWidget(self.irradiation_grid_edit, 0, 3)

        controls.addWidget(QLabel("Cooldown grid (s)", self), 1, 0)
        self.cooldown_grid_edit = QLineEdit("0,1800,7200,21600", self)
        self.cooldown_grid_edit.setObjectName("OptimizationCooldownGridEdit")
        controls.addWidget(self.cooldown_grid_edit, 1, 1)

        controls.addWidget(QLabel("Count grid (s)", self), 1, 2)
        self.count_grid_edit = QLineEdit("300,600,900,1800", self)
        self.count_grid_edit.setObjectName("OptimizationCountGridEdit")
        controls.addWidget(self.count_grid_edit, 1, 3)

        controls.addWidget(QLabel("Target nuclide", self), 2, 0)
        self.target_nuclide_edit = QLineEdit(self)
        self.target_nuclide_edit.setObjectName("OptimizationTargetNuclideEdit")
        self.target_nuclide_edit.setPlaceholderText("Optional FIM-C target")
        controls.addWidget(self.target_nuclide_edit, 2, 1)

        self.advanced_checkbox = QCheckBox("Enable advanced objectives", self)
        self.advanced_checkbox.setObjectName("OptimizationAdvancedCheck")
        controls.addWidget(self.advanced_checkbox, 2, 2, 1, 2)

        controls.addWidget(QLabel("LDRD sample", self), 3, 0)
        self.ldrd_sample_id_edit = QLineEdit(PHASE6_LDRD_DEFAULT_SAMPLE_ID, self)
        self.ldrd_sample_id_edit.setObjectName("OptimizationLDRDSampleIdEdit")
        controls.addWidget(self.ldrd_sample_id_edit, 3, 1)

        controls.addWidget(QLabel("LDRD output root", self), 3, 2)
        self.ldrd_output_root_edit = QLineEdit(
            str(phase6_ldrd_default_output_root(PHASE6_LDRD_DEFAULT_SAMPLE_ID)),
            self,
        )
        self.ldrd_output_root_edit.setObjectName("OptimizationLDRDOutputRootEdit")
        controls.addWidget(self.ldrd_output_root_edit, 3, 3)
        layout.addLayout(controls)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("Run Optimization", self)
        self.run_button.setObjectName("OptimizationRunButton")
        self.run_button.clicked.connect(self.run_optimization)
        button_row.addWidget(self.run_button)

        self.export_grid_button = QPushButton("Export Grid CSV", self)
        self.export_grid_button.setObjectName("OptimizationExportGridButton")
        self.export_grid_button.clicked.connect(self._export_grid_default)
        button_row.addWidget(self.export_grid_button)

        self.export_ffexp_button = QPushButton("Export .ffexp", self)
        self.export_ffexp_button.setObjectName("OptimizationExportFFEXPButton")
        self.export_ffexp_button.clicked.connect(self._export_ffexp_default)
        button_row.addWidget(self.export_ffexp_button)

        self.ldrd_worked_example_button = QPushButton("Run LDRD Worked Example", self)
        self.ldrd_worked_example_button.setObjectName("OptimizationLDRDWorkedExampleButton")
        self.ldrd_worked_example_button.clicked.connect(self.run_ldrd_worked_example)
        button_row.addWidget(self.ldrd_worked_example_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.summary = QLabel("Run an activity review before optimizing schedules.", self)
        self.summary.setObjectName("OptimizationSummaryLabel")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("OptimizationTabs")
        self.heatmap_table = QTableWidget(self)
        self.heatmap_table.setObjectName("OptimizationHeatmapTable")
        self.heatmap_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.heatmap_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.heatmap_table, "Heatmap")

        self.pareto_table = QTableWidget(self)
        self.pareto_table.setObjectName("OptimizationParetoTable")
        self.pareto_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pareto_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.pareto_table, "Pareto")

        self.recommendation_browser = QTextBrowser(self)
        self.recommendation_browser.setObjectName("OptimizationRecommendationBrowser")
        self.tabs.addTab(self.recommendation_browser, "Recommendation")
        layout.addWidget(self.tabs, 1)

    def _build_payload(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        activity_payload = _current_activity_payload(
            self.activity_review_provider,
            self.activity_results_provider,
        )
        if activity_payload is None:
            return None, None
        candidate_payload = build_difom_payload_from_activity_review(
            activity_payload,
            irradiation_grid_s=_parse_csv_floats(
                self.irradiation_grid_edit.text(),
                (1800.0, 3600.0, 7200.0, 14400.0),
            ),
            cooldown_grid_s=_parse_csv_floats(
                self.cooldown_grid_edit.text(),
                (0.0, 1800.0, 7200.0, 21600.0),
            ),
            count_grid_s=_parse_csv_floats(
                self.count_grid_edit.text(),
                (300.0, 600.0, 900.0, 1800.0),
            ),
            reference_irradiation_time_s=3600.0,
            flux_scale=1.0,
        )
        return activity_payload, candidate_payload

    def run_optimization(self) -> dict[str, Any] | None:
        activity_payload, candidate_payload = self._build_payload()
        if activity_payload is None or candidate_payload is None:
            self.summary.setText("Optimization is unavailable until activity results exist.")
            return None
        objective = str(self.objective_combo.currentData() or "di-fom")
        target_nuclide = self.target_nuclide_edit.text().strip() or None
        if objective == "di-fom":
            candidates, isotope_weights = parse_difom_sweep_payload(candidate_payload)
            output_payload = serialize_difom_ranking(
                rank_difom_schedules(candidates, isotope_weights=isotope_weights)
            )
        elif objective in {"fim-d", "fim-a", "fim-c"}:
            candidates, isotope_weights = parse_difom_sweep_payload(candidate_payload)
            output_payload = serialize_fim_ranking(
                rank_fim_schedules(
                    candidates,
                    isotope_weights=isotope_weights,
                    objective=objective,
                    target_nuclide=target_nuclide,
                    nuisance_variance_fraction=0.05,
                    regularization=1.0e-6,
                ),
                objective=objective,
            )
        elif objective == "mwdcs":
            candidates, isotope_weights = parse_mwdcs_sweep_payload(candidate_payload)
            output_payload = serialize_mwdcs_ranking(
                rank_mwdcs_schedules(
                    candidates,
                    isotope_weights=isotope_weights,
                    full_spectrum_mode=True,
                    overlap_penalty=0.1,
                )
            )
        elif objective == "bass-d":
            if not self.advanced_checkbox.isChecked():
                self.summary.setText("Enable advanced objectives to run BASS-D.")
                return None
            candidates, isotope_weights = parse_bassd_sweep_payload(candidate_payload)
            output_payload = serialize_bassd_ranking(
                rank_bassd_schedules(
                    candidates,
                    isotope_weights=isotope_weights,
                    dose_weight=0.02,
                    exploration_temperature=0.0,
                    seed=17,
                )
            )
        else:
            if not self.advanced_checkbox.isChecked():
                self.summary.setText("Enable advanced objectives to run STBD-MR.")
                return None
            candidates, isotope_weights = parse_stbdmr_sweep_payload(candidate_payload)
            output_payload = serialize_stbdmr_ranking(
                rank_stbdmr_schedules(
                    candidates,
                    isotope_weights=isotope_weights,
                    masking_regularization=0.1,
                    differentiable_graph_mode=True,
                    graph_temperature=2.0,
                )
            )

        output_payload["objective"] = objective
        support_artifacts = build_phase6_support_artifacts(
            activity_payload,
            output_payload,
            isotopes_of_interest=(),
        )
        self._last_activity_payload = activity_payload
        self._last_output_payload = output_payload
        self._last_support_artifacts = support_artifacts

        _fill_table(self.heatmap_table, support_artifacts["optimization_grid_rows"])
        pareto_rows = support_artifacts["recommended_schedule_rows"] or support_artifacts["optimization_grid_rows"]
        _fill_table(self.pareto_table, pareto_rows)

        top = (support_artifacts["recommended_schedule_rows"] or [{}])[0]
        self.recommendation_browser.setPlainText(
            "\n".join(
                [
                    f"Objective: {objective}",
                    f"Recommended label: {top.get('label', 'n/a')}",
                    f"Irradiation time (s): {top.get('irradiation_time_s', 'n/a')}",
                    f"Cooldown time (s): {top.get('cooldown_time_s', 'n/a')}",
                    f"Count time (s): {top.get('count_time_s', 'n/a')}",
                    f"Objective score: {top.get('objective_score', 'n/a')}",
                ]
            )
        )
        self.summary.setText(
            (
                f"Computed {len(support_artifacts['optimization_grid_rows'])} schedule rows for {objective}. "
                f"Prepared {len(support_artifacts['dose_endpoints_rows'])} dose endpoint(s)."
            )
        )
        return output_payload

    def export_grid_csv(self, path: str | Path) -> Path:
        if self._last_support_artifacts is None:
            if self.run_optimization() is None:
                raise ValueError("Optimization output is not available.")
        return _write_csv_rows(
            Path(path),
            self._last_support_artifacts["optimization_grid_rows"],
        )

    def export_ffexp(self, path: str | Path) -> Path:
        if self._last_output_payload is None or self._last_support_artifacts is None:
            if self.run_optimization() is None:
                raise ValueError("Optimization output is not available.")
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        write_ffexp_bundle(
            resolved,
            summary={
                "optimization_objective": self._last_output_payload.get("objective"),
                "optimization_candidate_count": len(
                    self._last_output_payload.get("ranked_candidates") or []
                ),
                "activity_complete": True,
                "comparison_ready": True,
            },
            metadata={
                "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "source": "gui_phase6_workspace",
            },
            activities=self._last_activity_payload,
            inventory={
                "time_series_rows": self._last_support_artifacts["inventory_timeseries_rows"],
                "activities_at_irradiation": self._last_support_artifacts[
                    "activities_at_irradiation_rows"
                ],
                "dose_endpoints": self._last_support_artifacts["dose_endpoints_rows"],
            },
            masking={
                "line_masking_results": self._last_support_artifacts["masking_candidates_rows"],
                "masking_isotope_ranking": self._last_support_artifacts["masking_isotope_rows"],
                "alternate_line_recommendations": self._last_support_artifacts["alternate_line_rows"],
            },
            optimization={
                "payload": self._last_output_payload,
                "optimization_grid": self._last_support_artifacts["optimization_grid_rows"],
                "recommended_schedules": self._last_support_artifacts[
                    "recommended_schedule_rows"
                ],
            },
            plot_manifest={"paths": []},
        )
        return resolved

    def _export_grid_default(self) -> None:
        self.export_grid_csv(Path.cwd() / "optimization_grid.csv")

    def _export_ffexp_default(self) -> None:
        self.export_ffexp(Path.cwd() / "phase6_bundle.ffexp")

    def run_ldrd_worked_example(self) -> Path | None:
        sample_id = self.ldrd_sample_id_edit.text().strip() or PHASE6_LDRD_DEFAULT_SAMPLE_ID
        output_text = self.ldrd_output_root_edit.text().strip()
        output_root = (
            Path(output_text)
            if output_text
            else phase6_ldrd_default_output_root(sample_id)
        )
        self.ldrd_output_root_edit.setText(str(output_root))
        try:
            summary_path = run_phase6_ldrd_worked_example(
                sample_id=sample_id,
                output_root=output_root,
            )
        except Exception as exc:  # pragma: no cover - exercised via GUI error reporting
            self.summary.setText(f"LDRD worked example failed: {exc}")
            return None
        self._last_worked_example_summary_path = summary_path
        self.summary.setText(
            "LDRD worked example completed for "
            f"{sample_id}. Summary: {summary_path.name}"
        )
        return summary_path

    def workflow_state(self) -> dict[str, Any]:
        return {
            "objective": str(self.objective_combo.currentData() or "di-fom"),
            "irradiation_grid_s": self.irradiation_grid_edit.text(),
            "cooldown_grid_s": self.cooldown_grid_edit.text(),
            "count_grid_s": self.count_grid_edit.text(),
            "target_nuclide": self.target_nuclide_edit.text(),
            "advanced_objectives": bool(self.advanced_checkbox.isChecked()),
            "ldrd_sample_id": self.ldrd_sample_id_edit.text(),
            "ldrd_output_root": self.ldrd_output_root_edit.text(),
            "current_tab": current_tab_label(self.tabs),
        }

    def apply_workflow_state(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        if "objective" in payload:
            set_combo_data(self.objective_combo, payload["objective"])
        if "irradiation_grid_s" in payload:
            self.irradiation_grid_edit.setText(str(payload["irradiation_grid_s"] or ""))
        if "cooldown_grid_s" in payload:
            self.cooldown_grid_edit.setText(str(payload["cooldown_grid_s"] or ""))
        if "count_grid_s" in payload:
            self.count_grid_edit.setText(str(payload["count_grid_s"] or ""))
        if "target_nuclide" in payload:
            self.target_nuclide_edit.setText(str(payload["target_nuclide"] or ""))
        if "advanced_objectives" in payload:
            self.advanced_checkbox.setChecked(bool(payload["advanced_objectives"]))
        if "ldrd_sample_id" in payload:
            self.ldrd_sample_id_edit.setText(str(payload["ldrd_sample_id"] or ""))
        if "ldrd_output_root" in payload:
            self.ldrd_output_root_edit.setText(str(payload["ldrd_output_root"] or ""))
        if "current_tab" in payload:
            set_tab_label(self.tabs, str(payload["current_tab"]))


class SecondIrradiationPlannerPanel(QWidget):
    """Qt surface for second-irradiation planning and comparison."""

    def __init__(
        self,
        *,
        mode_manager: ModeManager,
        activity_review_provider: ActivityReviewProvider,
        activity_results_provider: ActivityResultsProvider,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.mode_manager = mode_manager
        self.activity_review_provider = activity_review_provider
        self.activity_results_provider = activity_results_provider
        self._last_payload: dict[str, Any] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        intro = QLabel(
            (
                "Second Irradiation Planner compares candidate follow-on pulses using the "
                "current irradiation-time inventory and weighted nuclide objectives."
            ),
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QGridLayout()
        controls.addWidget(QLabel("First cooling (s)", self), 0, 0)
        self.first_cooling_spin = QDoubleSpinBox(self)
        self.first_cooling_spin.setObjectName("SecondIrradiationFirstCoolingSpin")
        self.first_cooling_spin.setRange(0.0, 1.0e8)
        self.first_cooling_spin.setDecimals(1)
        self.first_cooling_spin.setValue(3600.0)
        controls.addWidget(self.first_cooling_spin, 0, 1)

        controls.addWidget(QLabel("Second irradiation (s)", self), 0, 2)
        self.second_irradiation_spin = QDoubleSpinBox(self)
        self.second_irradiation_spin.setObjectName("SecondIrradiationDurationSpin")
        self.second_irradiation_spin.setRange(0.0, 1.0e8)
        self.second_irradiation_spin.setDecimals(1)
        self.second_irradiation_spin.setValue(1800.0)
        controls.addWidget(self.second_irradiation_spin, 0, 3)

        controls.addWidget(QLabel("Flux scales", self), 1, 0)
        self.flux_scales_edit = QLineEdit("0.75,1.0,1.35", self)
        self.flux_scales_edit.setObjectName("SecondIrradiationFluxScalesEdit")
        controls.addWidget(self.flux_scales_edit, 1, 1)

        controls.addWidget(QLabel("Duration factors", self), 1, 2)
        self.duration_factors_edit = QLineEdit("1.0,1.1,1.2", self)
        self.duration_factors_edit.setObjectName("SecondIrradiationDurationFactorsEdit")
        controls.addWidget(self.duration_factors_edit, 1, 3)

        controls.addWidget(QLabel("Cooling grid (s)", self), 2, 0)
        self.cooling_grid_edit = QLineEdit("600,900,1200", self)
        self.cooling_grid_edit.setObjectName("SecondIrradiationCoolingGridEdit")
        controls.addWidget(self.cooling_grid_edit, 2, 1)

        controls.addWidget(QLabel("Target weights", self), 2, 2)
        self.target_weights_edit = QLineEdit(self)
        self.target_weights_edit.setObjectName("SecondIrradiationTargetWeightsEdit")
        self.target_weights_edit.setPlaceholderText("Co-60:1.5,Mn-56:0.5")
        controls.addWidget(self.target_weights_edit, 2, 3)
        layout.addLayout(controls)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("Run Second Irradiation Plan", self)
        self.run_button.setObjectName("SecondIrradiationRunButton")
        self.run_button.clicked.connect(self.run_plan)
        button_row.addWidget(self.run_button)

        self.export_csv_button = QPushButton("Export Selected CSV", self)
        self.export_csv_button.setObjectName("SecondIrradiationExportCSVButton")
        self.export_csv_button.clicked.connect(self._export_default)
        button_row.addWidget(self.export_csv_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.summary = QLabel("Run an activity review before planning a second irradiation.", self)
        self.summary.setObjectName("SecondIrradiationSummaryLabel")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.table = QTableWidget(self)
        self.table.setObjectName("SecondIrradiationTable")
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        self.browser = QTextBrowser(self)
        self.browser.setObjectName("SecondIrradiationBrowser")
        layout.addWidget(self.browser, 1)

    def _parse_target_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for token in _parse_csv_strings(self.target_weights_edit.text()):
            if ":" not in token:
                continue
            key, value = token.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            weights[key] = float(value)
        return weights

    def run_plan(self) -> dict[str, Any] | None:
        payload = _current_activity_payload(
            self.activity_review_provider,
            self.activity_results_provider,
        )
        if payload is None:
            self.summary.setText("Second-irradiation planning is unavailable until activity results exist.")
            return None
        inventory_state = build_inventory_state_from_payload(
            payload,
            decay_source_id=str(payload.get("decay_source_id") or DEFAULT_DECAY_SOURCE_ID),
        )
        candidates = build_second_irradiation_candidates(
            flux_scales=_parse_csv_floats(self.flux_scales_edit.text(), (0.75, 1.0, 1.35)),
            duration_factors=_parse_csv_floats(self.duration_factors_edit.text(), (1.0, 1.1, 1.2)),
            cooling_times_s=_parse_csv_floats(self.cooling_grid_edit.text(), (600.0, 900.0, 1200.0)),
        )
        plan = plan_second_irradiation(
            inventory_state,
            first_cooling_time_s=float(self.first_cooling_spin.value()),
            second_irradiation_time_s=float(self.second_irradiation_spin.value()),
            target_weights=self._parse_target_weights(),
            candidates=candidates,
        )
        payload_out = serialize_second_irradiation_plan(plan)
        self._last_payload = payload_out
        _fill_table(self.table, payload_out["ranked_candidates"])
        selected = payload_out["selected_candidate"] or {}
        self.browser.setPlainText(
            "\n".join(
                [
                    f"Selected label: {selected.get('label', 'n/a')}",
                    f"Score: {selected.get('score', 'n/a')}",
                    f"Flux scale: {selected.get('flux_scale', 'n/a')}",
                    f"Duration factor: {selected.get('duration_factor', 'n/a')}",
                    "",
                    "Post-second-irradiation inventory:",
                    *[
                        (
                            f"{row['nuclide']}: activity={float(row['activity_bq']):.6g} Bq, "
                            f"weighted={float(row['weighted_activity']):.6g}"
                        )
                        for row in payload_out["selected_inventory_rows"]
                    ],
                ]
            )
        )
        self.summary.setText(
            (
                f"Evaluated {len(payload_out['ranked_candidates'])} second-irradiation candidate(s). "
                f"Selected {selected.get('label', 'n/a')}."
            )
        )
        return payload_out

    def export_selected_csv(self, path: str | Path) -> Path:
        if self._last_payload is None:
            if self.run_plan() is None:
                raise ValueError("Second-irradiation plan is not available.")
        return _write_csv_rows(Path(path), self._last_payload["selected_inventory_rows"])

    def _export_default(self) -> None:
        self.export_selected_csv(Path.cwd() / "second_irradiation_selected.csv")

    def workflow_state(self) -> dict[str, Any]:
        return {
            "first_cooling_s": float(self.first_cooling_spin.value()),
            "second_irradiation_s": float(self.second_irradiation_spin.value()),
            "flux_scales": self.flux_scales_edit.text(),
            "duration_factors": self.duration_factors_edit.text(),
            "cooling_grid_s": self.cooling_grid_edit.text(),
            "target_weights": self.target_weights_edit.text(),
        }

    def apply_workflow_state(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        if "first_cooling_s" in payload:
            self.first_cooling_spin.setValue(float(payload["first_cooling_s"]))
        if "second_irradiation_s" in payload:
            self.second_irradiation_spin.setValue(float(payload["second_irradiation_s"]))
        if "flux_scales" in payload:
            self.flux_scales_edit.setText(str(payload["flux_scales"] or ""))
        if "duration_factors" in payload:
            self.duration_factors_edit.setText(str(payload["duration_factors"] or ""))
        if "cooling_grid_s" in payload:
            self.cooling_grid_edit.setText(str(payload["cooling_grid_s"] or ""))
        if "target_weights" in payload:
            self.target_weights_edit.setText(str(payload["target_weights"] or ""))


__all__ = [
    "MaskingReviewPanel",
    "OptimizationWorkspacePanel",
    "SecondIrradiationPlannerPanel",
]
