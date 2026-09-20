import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from inference_common import compare, metrics, input_frame, verified, sha, load_model, normalize_saved_metrics, verify_source_match


def frame():
    return pd.DataFrame({'sample_id':['a','b','c','d'], 'sequence':['C','CC','CCC','CCCC'],
                         'label':[0,0,1,1], 'probability_bbb_plus':[0.1,0.7,0.6,0.9], 'prediction':[0,1,1,1]})


class ParityTests(unittest.TestCase):
    def test_flat_macro_confusion_matrix(self):
        value = {'accuracy': 0.75, 'tn': 1, 'fp': 1, 'fn': 0, 'tp': 2}
        result = normalize_saved_metrics(value)
        self.assertEqual(result['confusion_matrix'], {'tn': 1, 'fp': 1, 'fn': 0, 'tp': 2})
        self.assertNotIn('confusion_matrix', value)

    def test_source_match_without_separate_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            package, release = Path(d) / 'package', Path(d) / 'release'
            hashes = {}
            for rel in ('scripts/train_small_molecule_repro.py', 'plat_model/model.py'):
                for root in (package, release):
                    path = root / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('value = 1\n')
                hashes[rel] = sha(package / rel)
            verify_source_match(package, release, hashes)
            (release / 'plat_model/model.py').write_text('value = 2\n')
            with self.assertRaisesRegex(RuntimeError, 'different source'):
                verify_source_match(package, release, hashes)

    def test_source_file_set_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            package, release = Path(d) / 'package', Path(d) / 'release'
            (release / 'plat_model').mkdir(parents=True)
            (release / 'plat_model/extra.py').write_text('')
            with self.assertRaisesRegex(RuntimeError, 'file set differs'):
                verify_source_match(package, release, {})

    def test_identical(self):
        self.assertTrue(compare(frame(), frame())['pass'])

    def test_score_drift(self):
        right = frame()
        right.loc[0,'probability_bbb_plus'] += 0.01
        self.assertFalse(compare(frame(), right)['pass'])

    def test_tolerance(self):
        right = frame()
        right.loc[0,'probability_bbb_plus'] += 1e-8
        self.assertTrue(compare(frame(), right)['pass'])

    def test_class_flip_fails_even_for_tiny_score_difference(self):
        left, right = frame(), frame()
        left.loc[0, 'probability_bbb_plus'] = 0.5
        right.loc[0, 'probability_bbb_plus'] = 0.50000001
        right.loc[0, 'prediction'] = 1
        self.assertFalse(compare(left, right)['pass'])

    def test_order_fails(self):
        with self.assertRaises(RuntimeError):
            compare(frame(), frame().iloc[::-1])

    def test_nonfinite_fails(self):
        right = frame()
        right.loc[0,'probability_bbb_plus'] = np.nan
        with self.assertRaises(RuntimeError):
            compare(frame(), right)

    def test_metrics(self):
        m = metrics(frame())
        self.assertEqual(m['accuracy'], 0.75)
        self.assertEqual(m['confusion_matrix'], {'tn':1,'fp':1,'fn':0,'tp':2})

    def test_missing_label_allowed_but_duplicate_ids_not(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input.csv'
            frame().drop(columns='label').to_csv(p, index=False)
            self.assertEqual(len(input_frame(p)), 4)
            f = frame()
            f.loc[1,'sample_id'] = 'a'
            f.to_csv(p, index=False)
            with self.assertRaises(RuntimeError):
                input_frame(p)

    def test_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            p = root / 'x.py'
            p.write_text('first')
            h = {'x.py':sha(p)}
            self.assertEqual(verified(root, 'x.py', h), p)
            p.write_text('changed')
            with self.assertRaises(RuntimeError):
                verified(root, 'x.py', h)

    def test_strict_and_safe_loading(self):
        calls = {}
        class Model:
            def load_state_dict(self, state, strict):
                calls['strict'] = strict
            def eval(self):
                calls['eval'] = True
        def load(path, **kwargs):
            calls.update(kwargs)
            return {'model_config':{'model_name':'GraphTransformer'}, 'model_state_dict':{}}
        rt = SimpleNamespace(torch=SimpleNamespace(load=load))
        core = SimpleNamespace(instantiate_model=lambda *args: Model())
        load_model(core, rt, {'model_config':{'model_name':'GraphTransformer'}}, 'model.pt', 'cpu')
        self.assertTrue(calls['strict'])
        self.assertTrue(calls['weights_only'])
        self.assertTrue(calls['eval'])


if __name__ == '__main__':
    unittest.main()
