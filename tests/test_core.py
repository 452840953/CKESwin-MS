"""Synthetic CPU checks. No research data, checkpoints or saved predictions."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

class CoreTests(unittest.TestCase):
    def test_paper_configuration(self):
        from train_tri_modal_swin_fusion_v3_2 import TrainConfig
        cfg = TrainConfig(**json.loads((ROOT / 'configs/visual.json').read_text()))
        self.assertEqual((cfg.epochs, cfg.g_layers, cfg.fusion_layers, cfg.fusion_heads), (300, 4, 2, 4))
        self.assertEqual((cfg.aux_warmup_epochs, cfg.aux_decay_epochs, cfg.aux_min), (10, 100, 0.1))

    def test_gaussian_geometry(self):
        import torch
        from train_tri_modal_swin_fusion_v3_2 import gaussian_weight_map
        a = gaussian_weight_map(56, 56, torch.tensor([[28., 28.]]), torch.tensor([[2.]]), 'cpu')
        b = gaussian_weight_map(56, 56, torch.tensor([[28., 28.]]), torch.tensor([[5.]]), 'cpu')
        self.assertAlmostEqual(a.sum().item(), 1.0, places=5)
        self.assertAlmostEqual(b.sum().item(), 1.0, places=5)
        self.assertEqual(a.argmax().item(), 28 * 56 + 28)
        self.assertLess(b.max().item(), a.max().item())
        empty = gaussian_weight_map(7, 7, None, None, 'cpu')
        self.assertTrue(torch.allclose(empty, torch.full((7, 7), 1 / 49)))

    def test_graph_and_transformer(self):
        import torch
        from torch_geometric.data import Data, Batch
        from train_tri_modal_swin_fusion_v3_2 import GraphEncoder, TriModalFusionTransformer
        torch.manual_seed(1)
        graph = Data(x=torch.randn(3, 28), edge_index=torch.tensor([[0,1,1,2], [1,0,2,1]]), edge_attr=torch.randn(4, 9))
        batch = Batch.from_data_list([graph, graph.clone()])
        enc = GraphEncoder(28, 9, 128, 4).eval()
        nodes, pooled = enc(batch)
        self.assertEqual(tuple(nodes.shape), (6, 128))
        self.assertEqual(tuple(pooled.shape), (2, 128))
        fuser = TriModalFusionTransformer(256, 4, 2).eval()
        output = fuser(*[torch.randn(2, 256) for _ in range(3)])
        self.assertEqual(tuple(output.shape), (2, 256))
        self.assertTrue(torch.isfinite(output).all())

    def test_fusion_formula_and_gradient(self):
        import torch
        from evaluation.fusion import DiagStackingHead, _rf_to_logits
        head = DiagStackingHead(7)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 21)
        gi = torch.randn(3, 7)
        p = torch.softmax(torch.randn(3, 7), dim=1)
        actual = head(gi, _rf_to_logits(p))
        self.assertTrue(torch.allclose(actual, gi + p.log(), atol=1e-6))
        actual.square().mean().backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters()))

    def test_specimen_group_holdout(self):
        import numpy as np
        from chemistry.rf_helpers import train_eval_split_fixed_by_groups, assert_no_leak
        groups = np.repeat([f's{i}' for i in range(12)], 2)
        X = np.arange(48).reshape(24, 2)
        y = np.tile([0, 1], 12)
        Xtr, ytr, gtr, Xe, ye, cv = train_eval_split_fixed_by_groups(X, y, groups, {'s10','s11'}, print_info=False)
        self.assertEqual(len(Xtr), 20)
        self.assertEqual(len(Xe), 4)
        assert_no_leak(gtr, np.array(['s10','s11']), cv, Xtr, ytr)
        self.assertEqual(set(gtr).intersection({'s10','s11'}), set())

    def test_import_does_not_load_detectors(self):
        import dograph
        self.assertIsNone(dograph.SAM_PREDICTOR)
        self.assertIsNone(dograph.YOLO_MODEL)

if __name__ == '__main__':
    unittest.main()
