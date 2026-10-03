"""Stdlib tests: relocation, source identity and failure paths for real QG data."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

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
        for relative in [str(SCRIPT),str(MANIFEST)] + [r['path'] for r in manifest['resources']]:
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
