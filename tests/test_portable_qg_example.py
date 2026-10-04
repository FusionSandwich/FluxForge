"""Stdlib tests: relocation, source identity and failure paths for real QG data."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path('examples/RAFM_irradiation/run_portable_qg_example.py')
MANIFEST = Path('examples/RAFM_irradiation/quantumgold_reference/manifest.json')
spec = importlib.util.spec_from_file_location('portable_qg', REPO/SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class PortableQGExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='fluxforge portable QG ')
        cls.copy = Path(cls.tmp.name)/'relocated source with spaces'
        manifest = json.loads((REPO/MANIFEST).read_text(encoding='utf-8'))
        supplement_path = MANIFEST.parent/'supplemental_inputs/manifest.json'
        supplement = json.loads((REPO/supplement_path).read_text(encoding='utf-8'))
        for relative in [str(SCRIPT),str(MANIFEST),str(supplement_path)] + [r['path'] for r in manifest['resources']+supplement['resources']]:
            src, target = REPO/relative, cls.copy/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
        cls.manifest = manifest

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_relocated_isolated_interpreter_from_unrelated_cwd(self):
        result = subprocess.run([sys.executable,'-I',str(self.copy/SCRIPT),'--verify-only'],
                                cwd=self.tmp.name, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt['counts'],self.manifest['completeness'])
        self.assertEqual(receipt['original_report_lines_verified'],376)
        self.assertEqual(receipt['required_external_data_paths'],[])
        self.assertEqual(receipt['supplemental_audit']['older_report_ROI_rows'],12)
        self.assertEqual(receipt['supplemental_audit']['older_report_summaries'],6)
        self.assertEqual(receipt['dataset_identity']['status'],'SOURCE_BOUND')
        self.assertEqual(receipt['dataset_identity']['source_manifest_sha256'],
                         driver.hashlib.sha256((self.copy/MANIFEST).read_bytes()).hexdigest())
        self.assertEqual(receipt['engine_identity']['status'],'UNKNOWN')
        self.assertIsNone(receipt['engine_identity']['source_sha256'])

    def test_engine_content_identity_survives_relocation_and_exposes_tampering(self):
        roots = [Path(self.tmp.name)/name for name in ('engine A','engine B')]
        for root in roots:
            for name, content in [('pyproject.toml','[project]\nname="fluxforge"\n'),
                                  (str(SCRIPT),'print("driver")\n'),
                                  ('src/fluxforge/analysis.py','VALUE = 1\n')]:
                path = root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(content,encoding='utf-8')
        first, relocated = [driver.engine_identity(root) for root in roots]
        self.assertEqual(first['status'],'IDENTIFIED_BY_CONTENT')
        self.assertEqual(first['source_sha256'],relocated['source_sha256'])
        self.assertIsNone(first['revision'])
        (roots[1]/'src/fluxforge/analysis.py').write_text('VALUE = 2\n',encoding='utf-8')
        changed = driver.engine_identity(roots[1])
        self.assertNotEqual(first['source_sha256'],changed['source_sha256'])

    def test_declared_core_runtime_mismatch_is_explicit(self):
        root = Path(self.tmp.name)/'compatibility fixture'
        root.mkdir()
        (root/'pyproject.toml').write_text(
            '[project]\nrequires-python=">=3.11,<3.13"\n'
            'dependencies=["numpy>=1.26,<2.0", "scipy>=1.11"]\n',encoding='utf-8')
        with patch.object(driver.distribution_metadata,'version',
                          side_effect=lambda name: {'numpy':'2.5.1','scipy':'1.18.0'}[name]):
            receipt = driver.runtime_compatibility(root)
        self.assertEqual(receipt['status'],'INCOMPATIBLE')
        self.assertEqual(receipt['packages']['numpy']['status'],'INCOMPATIBLE')
        self.assertEqual(receipt['packages']['scipy']['status'],'COMPATIBLE')

    def test_recovered_percent_curve_rejects_invalid_brackets_and_extrapolation(self):
        path = Path(self.tmp.name)/'curve.csv'
        path.write_text('energy_keV,efficiency_fraction,original_efficiency_value,source_line,status\n'
                        '40,-0.01,-1,1,excluded_nonpositive\n'
                        '100,0.001,0.1,2,working_percent_assumption\n'
                        '200,0.002,0.2,3,working_percent_assumption\n'
                        '300,-0.01,-1,4,excluded_nonpositive\n'
                        '400,0.003,0.3,5,working_percent_assumption\n'
                        '500,0.004,0.4,6,working_percent_assumption\n',encoding='utf-8')
        curve = driver.SouthWorkingCurve(path)
        self.assertAlmostEqual(curve.efficiency(150),0.0015)
        for energy in (50,300,350,550):
            with self.subTest(energy=energy),self.assertRaises(ValueError):
                curve.efficiency(energy)
        self.assertEqual(curve.to_dict()['positive_bracket_segments'],2)
        path.write_text(path.read_text().replace('100,0.001,0.1','100,0.1,0.1'),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'percent interpretation'):
            driver.SouthWorkingCurve(path)

    def test_raw_mode_excludes_near_contact_and_ti_produced_sc48(self):
        output = Path(self.tmp.name)/'must not exist'
        with self.assertRaisesRegex(ValueError,'near-contact'):
            driver.run_raw_comparison(self.copy,output,'Fe-Cd-RAFM-1')
        with self.assertRaisesRegex(ValueError,'Sc-48'):
            driver.run_raw_comparison(self.copy,output,'Ti-RAFM-1')
        self.assertFalse(output.exists())

    def test_south_native_background_is_source_bound_and_changes_raw_result(self):
        north = driver.run_raw_comparison(
            REPO, Path(self.tmp.name)/'north_background', 'Co-Cd-RAFM-1',
            background_mode='north_historical')
        south = driver.run_raw_comparison(
            REPO, Path(self.tmp.name)/'south_background', 'Co-Cd-RAFM-1',
            background_mode='south_native')
        self.assertEqual(south['background_details']['detector'], 'South')
        self.assertEqual(south['background_details']['source_sha256'],
                         '96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344')
        self.assertEqual(south['background_details']['live_time_s'], 14400.0)
        self.assertEqual(south['background_details']['total_counts'], 543427)
        self.assertEqual(south['background_details']['QG_background_match'], 'UNKNOWN')
        self.assertEqual(north['background_details']['detector'], 'North')
        self.assertNotEqual(north['background_source_sha256'], south['background_source_sha256'])
        self.assertNotEqual(north['selected_raw_activities']['Co60']['activity_bq'],
                            south['selected_raw_activities']['Co60']['activity_bq'])
        self.assertEqual(north['QG_reference_activities_Bq'], south['QG_reference_activities_Bq'])

    def test_south_native_source_tamper_is_rejected(self):
        path = self.copy/'examples/RAFM_irradiation/quantumgold_reference/supplemental_inputs/South 4hr Background Terminal.ANS'
        original = path.read_bytes()
        try:
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'Supplemental source hash/size mismatch'):
                driver.verify_inputs(self.copy, self.copy/MANIFEST)
        finally:
            path.write_bytes(original)

    def test_south_background_mode_cannot_be_silently_ignored_by_replay(self):
        result = subprocess.run([sys.executable, str(REPO/SCRIPT), '--verify-only',
                                 '--background-mode', 'south_native'],
                                capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires --raw-sample', result.stderr)

    def test_recovered_curve_must_match_original_percent_export(self):
        original = Path(self.tmp.name)/'original_export.csv'
        derived = Path(self.tmp.name)/'derived_curve.csv'
        original.write_text('Energy, Efficiency\n100,0.1\n200,0.2\n',encoding='utf-8')
        derived.write_text('energy_keV,efficiency_fraction,original_efficiency_value,source_line,status\n'
                           '100,0.001,0.1,2,working_percent_assumption\n'
                           '200,0.002,0.2,3,working_percent_assumption\n',encoding='utf-8')
        driver.SouthWorkingCurve(derived).verify_source_export(original)
        derived.write_text(derived.read_text().replace('200,0.002,0.2','200,0.003,0.3'),
                           encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'source-bound original'):
            driver.SouthWorkingCurve(derived).verify_source_export(original)

    def check_bad_manifest(self, changed):
        tmp_manifest = Path(self.tmp.name)/'bad_manifest.json'
        tmp_manifest.write_text(json.dumps(changed),encoding='utf-8')
        with self.assertRaises((ValueError,FileNotFoundError)):
            driver.verify_inputs(self.copy,tmp_manifest)

    def test_misbound_report_rejected(self):
        changed = json.loads(json.dumps(self.manifest))
        changed['measurements'][0]['files']['QG_report'] = changed['measurements'][1]['files']['QG_report']
        self.check_bad_manifest(changed)

    def test_duplicate_measurement_rejected(self):
        changed = json.loads(json.dumps(self.manifest))
        changed['measurements'][1]['measurement_id'] = changed['measurements'][0]['measurement_id']
        self.check_bad_manifest(changed)

    def test_metadata_only_routing_and_timing_changes_rejected(self):
        for field,value in [('cohort','RAFM1'),('workflow_stem','another_sample'),
                            ('raw_workflow_role','ASC_replay_wrong'),('sample_kind','wrong')]:
            changed = json.loads(json.dumps(self.manifest))
            changed['measurements'][0][field] = value
            with self.subTest(field=field):
                self.check_bad_manifest(changed)
        changed = json.loads(json.dumps(self.manifest))
        changed['measurements'][0]['timeline']['acquisition']['live_time_s'] = 1
        self.check_bad_manifest(changed)

    def test_missing_and_changed_source_rejected(self):
        pin = self.manifest['resources'][0]
        path = self.copy/pin['path']
        original = path.read_bytes()
        try:
            path.write_bytes(original[:-1]+bytes([original[-1]^1]))
            with self.assertRaises(ValueError):
                driver.verify_inputs(self.copy,self.copy/MANIFEST)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                driver.verify_inputs(self.copy,self.copy/MANIFEST)
        finally:
            path.write_bytes(original)

    def test_absolute_and_escape_paths_rejected(self):
        for name in ('../external.dat','C:/Users/example/data.txt','/tmp/data','data\\file'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                driver.bound_path(self.copy,name)

    def test_missing_sources_and_cohorts_are_explicit(self):
        missing_reports = [r['measurement_id'] for r in self.manifest['measurements'] if r['missing_report']]
        self.assertEqual(missing_reports,['RAFM-A-2hr'])
        missing_asc = {r['measurement_id'] for r in self.manifest['measurements'] if not r['files']['ASC']}
        self.assertEqual(missing_asc,{'Cu-Cd-RAFM-1','Fe-Cd-RAFM-1'})
        cohorts = {r['measurement_id']:r['physical_specimen_id'] for r in self.manifest['measurements']}
        self.assertNotEqual(cohorts['RAFM-A-15d'],cohorts['RAFM-A-24hr'])

    def test_monitor_output_labels_do_not_become_older_RAFM1(self):
        target = Path(self.tmp.name)/'label_reconciliation_fixture'
        folder = target/'qg_report_replay'
        folder.mkdir(parents=True,exist_ok=True)
        path = folder/'summary.json'
        value = {'samples':[{'sample_id':'Cu-Cd-RAFM-1_25cm','sample_group':'RAFM1',
                             'activity_Bq':247,'timing':{'sample_group':'RAFM1','live_s':3600}},
                            {'sample_id':'RAFM1_Long_70d_EOI','sample_group':'RAFM1','activity_Bq':42}]}
        path.write_text(json.dumps(value),encoding='utf-8')
        report = folder/'Cu-Cd-RAFM-1_25cm_comparison.txt'
        report.write_text('Sample group: RAFM1\nNet counts: 247\n',encoding='utf-8')
        driver.label_replay_outputs(target,self.manifest)
        fixed = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(fixed['samples'][0]['sample_group'],'flux_wires')
        self.assertEqual(fixed['samples'][0]['legacy_sample_group'],'RAFM1')
        self.assertEqual(fixed['samples'][0]['timing']['sample_group'],'flux_wires')
        self.assertEqual(fixed['samples'][0]['activity_Bq'],247)
        self.assertEqual(fixed['samples'][1],value['samples'][1])
        self.assertIn('Sample group: flux_wires',report.read_text(encoding='utf-8'))
        self.assertIn('Legacy sample group: RAFM1',report.read_text(encoding='utf-8'))
        self.assertIn('Net counts: 247',report.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
