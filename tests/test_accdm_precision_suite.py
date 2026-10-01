import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from campaigns.accdm import precision_suite as suite
from cluster.campaign_manager import QueueSnapshot


class PrecisionSuiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='lya precision,')
        self.root=Path(self.tmp.name)
        self.args=SimpleNamespace(suite=self.root/'suite',points=suite.POINTS,
            class_source=Path('/class'),account='pl0503-01',cpus=4,cpu_budget=40,
            memory='8G',walltime='24:00:00',partition='standard',
            variants=list(suite.VARIANTS),software_test_quality=None)
        self.runtime={'python':sys.executable,'classy_sha256':'abc',
            'declared_source':{'directory':'/class','commit':'123','branch':'accDM_refactor'}}
        self.mock=patch.object(suite,'class_runtime',return_value=self.runtime)
        self.mock.start()
        self.config=suite.build(self.args)

    def tearDown(self):
        self.mock.stop()
        self.tmp.cleanup()

    def test_matrix_changes_one_precision_axis_at_a_time(self):
        self.assertEqual(len(self.config['cases']),36)
        self.assertEqual(self.config['resources']['array_concurrency'],10)
        base=suite.read_json(self.args.suite/'inputs/baseline_base.json')['class_params']
        expected={'q100':{'accdm_q_bins_per_decade'},'background80k':{'background_Nloga'},
                  'birthtol1e8':{'accdm_q_number_tol'}}
        for name,keys in expected.items():
            p=suite.read_json(self.args.suite/f'inputs/{name}_base.json')['class_params']
            self.assertEqual({k for k in p if p[k]!=base[k]},keys)
        self.assertEqual(suite.read_json(self.args.suite/'inputs/joint_base.json'),
                         suite.read_json(self.args.suite/'inputs/joint_spt_base.json'))
        self.assertEqual(self.config['variants']['joint_spt']['spt_quality'],'precision')
        self.assertEqual(self.config['variants']['baseline']['spt_quality'],'production')

    @patch.object(suite.subprocess,'run')
    def test_dry_run_caps_array_cpu_budget_and_never_submits(self,run):
        cmd=suite.submit(self.args.suite,dry_run=True)
        self.assertIn('--cpus-per-task=4',cmd)
        self.assertTrue(next(x for x in cmd if x.startswith('--array=')).endswith('%10'))
        run.assert_not_called()
        self.assertFalse((self.args.suite/'submission.json').exists())

    def test_changed_precision_suite_is_immutable(self):
        before=(self.args.suite/'suite.json').read_bytes()
        self.args.cpus=2
        with self.assertRaisesRegex(ValueError,'settings changed'):
            suite.build(self.args)
        self.assertEqual((self.args.suite/'suite.json').read_bytes(),before)

    def test_variant_order_cannot_change_running_array_indices(self):
        self.args.variants.reverse()
        with self.assertRaisesRegex(ValueError,'settings changed'):
            suite.build(self.args)

    @patch.object(suite.subprocess,'run')
    def test_submission_passes_paths_as_environment_and_records_id(self,run):
        run.return_value=subprocess.CompletedProcess([],0,'12345;eagle\n','')
        suite.submit(self.args.suite)
        self.assertEqual(run.call_args.kwargs['env']['SUITE_DIR'],str(self.args.suite))
        self.assertEqual(suite.read_json(self.args.suite/'submission.json')['job_id'],'12345')

    @patch.object(suite,'query_queue',return_value=QueueSnapshot(frozenset({'123_5'}),1))
    @patch.object(suite.subprocess,'run')
    def test_active_array_is_not_submitted_twice(self,run,queue):
        (self.args.suite/'submission.json').write_text(json.dumps({'job_id':'123'}))
        with self.assertRaisesRegex(RuntimeError,'still active'):
            suite.submit(self.args.suite)
        run.assert_not_called()

    @patch.object(suite.subprocess,'run')
    def test_failed_theory_blocks_fit(self,run):
        run.return_value=subprocess.CompletedProcess([],2)
        self.assertEqual(suite.run_point(self.args.suite,0),2)
        self.assertEqual(run.call_count,1)
        self.assertEqual(run.call_args.kwargs['env']['THEORY_THREADS'],'4')
        self.assertEqual(suite.read_json(self.args.suite/'timings/0_theory.json')['exit_code'],2)


if __name__=='__main__':
    unittest.main()
