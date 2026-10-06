"""Full CPU forward smoke check with in-memory random initialization.

The checkpoint reader is supplied the just-created random backbone state in
memory. No pretrained/research weights are read, downloaded or saved. This checks
architecture wiring, not classification quality or historical checkpoint loading.
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

class ModelSmoke(unittest.TestCase):
    def test_forward_and_parameter_count(self):
        import torch
        from torch_geometric.data import Data, Batch
        import train_tri_modal_swin_fusion_v3_2 as m
        torch.set_num_threads(2)
        torch.manual_seed(7)
        original_create = m.timm.create_model
        original_isfile = os.path.isfile
        marker = '__synthetic_in_memory_checkpoint__'
        cache = {}

        def create(*args, **kwargs):
            self.assertFalse(kwargs.get('pretrained', True))
            backbone = original_create(*args, **kwargs)
            cache['state'] = backbone.state_dict()
            return backbone

        cfg = m.TrainConfig(swin_ckpt_path=marker)
        with patch.object(m.timm, 'create_model', side_effect=create), \
             patch.object(m.os.path, 'isfile', side_effect=lambda p: str(p) == marker or original_isfile(p)), \
             patch.object(m.torch, 'load', side_effect=lambda *a, **kw: cache['state']):
            model = m.TriModalSwinFusionModel(28, 9, 7, cfg).eval()
        # The paper's checkpoint tensor count includes 2,026 buffer elements.
        self.assertEqual(sum(p.numel() for p in model.parameters()), 95724822)
        unique_tensors = {id(t): t for t in model.state_dict(keep_vars=True).values()}
        # state_dict excludes non-persistent Swin attention masks and indices.
        self.assertEqual(sum(t.numel() for t in unique_tensors.values()), 95726848)
        graph = Data(x=torch.randn(3, 28), edge_index=torch.tensor([[0,1,1,2], [1,0,2,1]]),
                     edge_attr=torch.randn(4, 9), node_pos_feat=torch.tensor([[12.,12.],[24.,24.],[40.,40.]]),
                     node_r_feat=torch.full((3, 1), 2.0))
        batch = Batch.from_data_list([graph])
        with torch.no_grad():
            scores, detail = model(batch, torch.randn(1, 3, 224, 224), return_details=True)
        self.assertEqual(tuple(scores.shape), (1, 7))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual([tuple(s[-2:]) for s in detail['debug']['stages']], [(56,56),(28,28),(14,14),(7,7)])
        for key in ['z_G', 'z_I', 'z_R']:
            self.assertEqual(tuple(detail[key].shape), (1, 256))

if __name__ == '__main__':
    unittest.main()
