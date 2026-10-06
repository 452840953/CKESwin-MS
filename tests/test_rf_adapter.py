"""RF CLI integration using generated toy inputs in a temporary directory."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class RFAdapter(unittest.TestCase):
    def test_fit_export_and_holdout(self):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory(prefix='ckeswin_rf_test_') as tmp:
            base = Path(tmp)
            rows, heldout = [], []
            for species in range(7):
                for specimen in range(4):
                    sid = f'class{species}_specimen{specimen}'
                    if specimen == 3:
                        heldout.append(sid)
                    for repeat in range(2):
                        rows.append([f'class{species}', sid, repeat, *rng.normal(species, 0.1, 4)])
            pd.DataFrame(rows, columns=['label','specimen','repeat','f0','f1','f2','f3']).to_csv(base/'spectra.csv',index=False)
            (base/'test.txt').write_text('\n'.join(heldout))
            cfg = json.loads((ROOT/'configs/rf.json').read_text())
            cfg.update(n_estimators=8, max_depth=3, n_jobs=1)
            (base/'rf.json').write_text(json.dumps(cfg))
            run = subprocess.run([sys.executable, '-B', str(ROOT/'run.py'), 'train-rf',
                '--csv', str(base/'spectra.csv'), '--test-list', str(base/'test.txt'),
                '--config', str(base/'rf.json'), '--output', str(base/'out'), '--expected-features','4'],
                capture_output=True, text=True, encoding='utf-8', errors='replace')
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            predictions = pd.read_csv(base/'out/ms_probabilities.csv')
            self.assertEqual(len(predictions), 56)
            self.assertTrue(np.allclose(predictions[[f'p{i}' for i in range(7)]].sum(axis=1), 1))
            split = json.loads((base/'out/split_indices_fixed_eval.json').read_text())
            self.assertEqual(len(split['eval_idx']), 14)
            self.assertFalse({rows[i][1] for i in split['train_pool_idx']} & set(heldout))
            mapping = json.loads((base/'out/probability_class_order.json').read_text())
            self.assertEqual(mapping, {f'p{i}':f'class{i}' for i in range(7)})

if __name__ == '__main__':
    unittest.main()
