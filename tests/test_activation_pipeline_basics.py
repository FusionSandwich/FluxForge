from __future__ import annotations

from fluxforge.workflows.activation_pipeline import (
    ALARAConfig,
    ActivationComparison,
    ActivationResult,
    PipelineResult,
)


def test_alara_config_find_executable_returns_string_or_none():
    cfg = ALARAConfig(executable="python")
    found = cfg.find_executable()
    assert found is None or isinstance(found, str)


def test_activation_comparison_and_pipeline_summary():
    calc = ActivationResult(
        isotope="Co60", activity_Bq=100.0, activity_unc=10.0, source="ALARA"
    )
    meas = ActivationResult(
        isotope="Co60", activity_Bq=95.0, activity_unc=8.0, source="measured"
    )
    comp = ActivationComparison(isotope="Co60", calculated=calc, measured=meas)

    assert comp.c_over_e > 0
    assert comp.c_over_e_unc >= 0

    result = PipelineResult(
        success=True,
        activation_results=[calc],
        comparisons=[comp],
        execution_time_s=1.23,
        messages=["ok"],
    )

    text = result.summary()
    assert "SUCCESS" in text
    assert "Co60" in text
    assert result.mean_c_over_e > 0
