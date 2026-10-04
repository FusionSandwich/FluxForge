from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from fluxforge.analysis.flux_unfold import FluxWireReaction
from fluxforge.examples.rafm_workflow import (
    load_rafm_example_metadata, resolve_measurement_timing,
    reaction_rows_to_dicts, reaction_from_row,
    optional_export_number,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('integrated_methods_test', ROOT/'examples/RAFM_irradiation/run_integrated_qg_methods.py')
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_csv_nulls_remain_missing_and_numeric_strings_remain_numeric():
    assert optional_export_number('') is None
    assert optional_export_number(None) is None
    assert optional_export_number('0') == 0
    assert optional_export_number('1e-12') == 1e-12
    with pytest.raises(ValueError):
        optional_export_number('nan')


def test_cu_25cm_alias_binds_to_the_source_monitor_schedule():
    metadata = load_rafm_example_metadata(ROOT/'examples/RAFM_irradiation/quantumgold_reference/runtime')
    timing = resolve_measurement_timing('Cu-RAFM-1_25cm', None, metadata)
    original = metadata.sample_schedule['flux_wires']['CU-RAFM-1']
    assert timing.sample_group == 'flux_wires'
    assert timing.compare_eoi
    assert timing.irradiation_time_s == original['irradiation_seconds']
    assert timing.decay_time_s == original['cooldown_seconds']
    assert timing.irradiation_end.isoformat() == original['irradiation_end']


def test_conflicting_schedule_aliases_are_rejected():
    metadata = load_rafm_example_metadata(ROOT/'examples/RAFM_irradiation/quantumgold_reference/runtime')
    metadata.sample_schedule['flux_wires']['Cu-RAFM-1_25cm'] = dict(
        metadata.sample_schedule['flux_wires']['CU-RAFM-1'], cooldown_seconds=1)
    with pytest.raises(ValueError, match='Ambiguous'):
        resolve_measurement_timing('Cu-RAFM-1_25cm', None, metadata)


def test_unavailable_eoi_exports_null_not_zero_and_stays_excluded():
    reaction = FluxWireReaction('Ti-RAFM-1', 'Ti-50(n,g)Ti-51', 'Ti51', 0,
        rate_note='no end-of-irradiation activity (decay timing missing or non-finite)')
    row = reaction_rows_to_dicts([reaction])[0]
    assert row['rate_status'] == 'UNAVAILABLE'
    assert row['reaction_rate'] is None and row['activity_bq'] is None
    rebuilt = reaction_from_row(row)
    assert rebuilt.reaction_rate == 0 and rebuilt.rate_note == reaction.rate_note
    assert reaction_rows_to_dicts([rebuilt])[0]['reaction_rate'] is None


def test_available_zero_does_not_become_missing():
    reaction = FluxWireReaction('known', 'known', 'known', 0, rate_note='valid measured zero')
    row = reaction_rows_to_dicts([reaction])[0]
    assert row['reaction_rate'] == 0 and row['rate_status'] == 'DIAGNOSTIC'


def test_combination_consumes_physical_line_activities_not_report_targets(tmp_path):
    folder = tmp_path/'raw_replay'/'analysis_json'
    folder.mkdir(parents=True)
    payload = dict(sample_group='flux_wires', sample_id='Co-Cd',
        QG_reference_activity=99999, comparison_net_counts=99999,
        isotopes={'Co60':{'activity_reference':'count_average_live_normalized', 'single_peak_activity_diagnostics':[
            dict(energy_keV=1173.2,line_activity_bq=10,line_activity_unc_bq=1),
            dict(energy_keV=1332.5,line_activity_bq=20,line_activity_unc_bq=1)]}})
    (folder/'sample.json').write_text(json.dumps(payload))
    receipt = driver.current_line_combinations(tmp_path, 'engine-test')
    inverse = next(m for m in receipt[0]['methods'] if m['method']=='inverse_variance')
    assert inverse['activity_bq'] == pytest.approx(15)
    assert inverse['status'] == 'conditional'


def test_unknown_line_activity_reference_is_not_invented(tmp_path):
    folder = tmp_path/'raw_replay'/'analysis_json'
    folder.mkdir(parents=True)
    (folder/'sample.json').write_text(json.dumps(dict(sample_group='flux_wires', sample_id='Co',
        isotopes={'Co60':{'single_peak_activity_diagnostics':[
            dict(energy_keV=1173.2,line_activity_bq=10,line_activity_unc_bq=1)]}})))
    row = driver.current_line_combinations(tmp_path, 'engine-test')[0]
    assert row['methods'] == []
    assert row['status'] == 'EXCLUDED_UNKNOWN_OR_INCOMPATIBLE_ACTIVITY_REFERENCE'


@pytest.mark.parametrize('values,expected_count,expected_error', [
    (('', ''), 0, None), (('10', '5'), 1, 1.0), (('0', '5'), 1, 1.0),
])
def test_branch_comparison_distinguishes_missing_from_agreement(tmp_path, values, expected_count, expected_error):
    from fluxforge.examples.rafm_workflow import compare_rafm_completion_results
    for name, value in zip(('raw', 'qg'), values):
        folder = tmp_path/name/'tables'
        folder.mkdir(parents=True)
        (folder/'flux_wire_reaction_rates.csv').write_text(
            'sample_id,reaction_id,activity_bq,reaction_rate\nA,R,' + value + ',' + value + '\n')
    summary = compare_rafm_completion_results(tmp_path/'raw', tmp_path/'qg', tmp_path/'comparison')
    assert summary['matched_reactions'] == 1
    assert summary['valid_rate_comparisons'] == summary['valid_activity_comparisons'] == expected_count
    assert summary['median_abs_rate_rel_error'] == summary['max_abs_activity_rel_error'] == expected_error
    if not expected_count:
        assert summary['rate_comparison_status'] == 'UNAVAILABLE'
        assert 'UNAVAILABLE' in (tmp_path/'comparison/branch_comparison.md').read_text()


@pytest.mark.parametrize('method,expected', [
    ('qg', 'qg_report_measurement_date'),
    ('iec_tiered', 'count_average_live_normalized'),
])
def test_actual_flux_wire_recombination_preserves_time_reference(tmp_path, monkeypatch, method, expected):
    import fluxforge.examples.rafm_workflow as workflow
    from fluxforge.io.flux_wire import read_raw_asc, read_processed_txt
    metadata = load_rafm_example_metadata(ROOT/'examples/RAFM_irradiation/quantumgold_reference/runtime')
    metadata.config['flux_wire_counting_method'] = method
    metadata.config['qg_report_activity_includes_count_decay'] = False
    paths = workflow.default_paths(ROOT/'examples/RAFM_irradiation', results_root=tmp_path/'results')
    raw = paths.raw_root/'flux_wires/Co-Cd-RAFM-1_25cm.ASC'
    report = paths.qg_root/'flux_wires/Co-Cd-RAFM-1_25cm.txt'
    background = read_raw_asc(paths.background_path, profile_name=metadata.config['profile_name']).spectrum
    seen = {}
    aggregate = workflow.aggregate_isotope_results
    def capture(*args, **kwargs):
        seen.update(kwargs['count_real_time_by_isotope'])
        return aggregate(*args, **kwargs)
    monkeypatch.setattr(workflow, 'aggregate_isotope_results', capture)
    artifact = workflow.analyze_flux_wire_sample(raw, metadata, paths,
        workflow.ensure_results_tree(paths.results_root), background, report,
        workflow.normalize_pairing_key(raw.stem, metadata.pairing_aliases))
    assert artifact['isotopes']['Co60']['activity_reference'] == expected
    if method == 'qg':
        assert seen['Co60'] == read_processed_txt(report).real_time
        assert seen['Co60'] != read_raw_asc(raw).real_time
    else:
        assert seen == {}


def test_native_only_count_is_read_without_asc_and_wrong_geometry_is_excluded(tmp_path):
    portable = driver.load_driver()
    checked = portable.verify_inputs(ROOT, portable.MANIFEST_PATH)
    rows = driver.native_only_controls(portable, checked, tmp_path)
    by_id = {r['measurement_id']:r for r in rows}
    assert by_id['Cu-Cd-RAFM-1']['status']=='CONDITIONAL_NATIVE_RAW_DIAGNOSTIC'
    assert by_id['Fe-Cd-RAFM-1']['status']=='EXCLUDED_GEOMETRY'
    assert all(r['creates_ASC'] is False and r['reaction_rate'] is None for r in rows)
    assert not list(tmp_path.glob('*.ASC'))
