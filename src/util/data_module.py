# util/data_module.py
# -*- coding: utf-8 -*-
from typing import Tuple, Dict, Optional
import os
import torch
import json
from torch.utils.data import DataLoader, Subset

from dograph import WoodCellsGraphDataset
from FusionDataset import FusionDataset, fusion_collate
from util.fusion_debug import clean_graph_dataset
from util.split_util import split_dataset_by_specimens, make_fusion_datasets
from util.fusion_utils import _log_split_report, check_rf_coverage
from create_graph.config import SAVEROOT, IMAGEROOT

from torch.utils.data import Dataset

class _GraphTransformWrapper(Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        g = self.base[idx]          # 取出 Data（来自 Subset 或 list）
        return self.transform(g)    # 强制应用你的 _standardize_graph
    
class WoodFusionDataModule:
    """
    负责：
      1) 加载/清洗 PyG 图数据集
      2) RF 覆盖率检查
      3) 按 specimen 固定划分 train/val/test（读取 cfg.test_file）
      4) 基于 PyG 子集构造 FusionDataset（三元组：graph, image, rf）
      5) 构建 DataLoader
    """
    def __init__(self, cfg, device, logger):
        self.cfg = cfg
        self.device = device
        self.logger = logger

        # 这些属性会在 setup() 后可用
        self.pyg_ds = None
        self.clean_stats = None
        self.train_idx = self.val_idx = self.test_idx = None
        self.idx_to_class: Optional[Dict[int, str]] = None

        self.pyg_train = self.pyg_val = self.pyg_test = None
        self.ds_train = self.ds_val = self.ds_test = None
        self.loader_train = self.loader_val = self.loader_test = None

    # -------- 主流程 --------
    def setup(self):
        # 1) 读取 PyG 数据集
        self.pyg_ds = WoodCellsGraphDataset(root=SAVEROOT, data_root=IMAGEROOT, verbose=True)

        # 2) 清洗（与你原脚本保持一致）
        self.pyg_ds, self.clean_stats = clean_graph_dataset(
            self.pyg_ds, self.logger, mode="strict_drop", fields_strict=("x",), max_log=20
        )

        # 3) RF 覆盖率检查（方便提前发现 specimen 对不上等问题）
        check_rf_coverage(
            self.pyg_ds,
            self.cfg.rf_csv,
            self.cfg.rf_num_classes,
            key_col="specimen_id",
            max_show=30
        )

        # 4) 固定划分（使用你已有的 split 文件）
        if not os.path.exists(self.cfg.test_file):
            raise FileNotFoundError(f"未找到 test_file: {self.cfg.test_file}")

        self.train_idx, self.val_idx, self.test_idx, self.idx_to_class = split_dataset_by_specimens(
            self.pyg_ds, self.cfg.test_file, self.cfg.out_dir, self.logger
        )
        _log_split_report(self.pyg_ds, self.train_idx, self.val_idx, self.logger)
        # ===== 4.5) 拟合并设置图特征标准化（仅用 train 拟合） =====  # NEW
        with torch.no_grad():
            # --- x: 节点特征 ---
            sx = ssx = nx = None
            Dx = None
            for i in self.train_idx:
                g = self.pyg_ds[i]
                if getattr(g, "x", None) is None or g.x.numel() == 0:
                    continue
                x = g.x.float()
                if Dx is None:
                    Dx = x.size(1)
                    sx = torch.zeros(Dx)
                    ssx = torch.zeros(Dx)
                sx += x.sum(dim=0)
                ssx += (x * x).sum(dim=0)
                nx = (nx or 0) + x.size(0)
            if nx is not None and nx > 0:
                mean_x = sx / nx
                var_x = (ssx / nx) - mean_x ** 2
                std_x = torch.sqrt(var_x.clamp_min(1e-12))
            else:
                mean_x = std_x = None
            
            # --- edge_attr: 边特征（可选，如不存在则跳过） ---
            se = sse = ne = None
            De = None
            for i in self.train_idx:
                g = self.pyg_ds[i]
                ea = getattr(g, "edge_attr", None)
                if ea is None or ea.numel() == 0:
                    continue
                ea = ea.float()
                if De is None:
                    De = ea.size(1)
                    se = torch.zeros(De)
                    sse = torch.zeros(De)
                se += ea.sum(dim=0)
                sse += (ea * ea).sum(dim=0)
                ne = (ne or 0) + ea.size(0)
            if ne is not None and ne > 0:
                mean_e = se / ne
                var_e = (sse / ne) - mean_e ** 2
                std_e = torch.sqrt(var_e.clamp_min(1e-12))
            else:
                mean_e = std_e = None
        
        # 设置 transform：每次 __getitem__ 都会用到同一套参数
        def _standardize_graph(g):
            g = g.clone()
            
            # 统一标签
            if getattr(g, "y", None) is not None:
                g.y = g.y.view(-1).long()
            
            # —— 先备份 raw（标准化之前）——
            x0 = g.x.detach().clone() if getattr(g, "x", None) is not None and g.x.numel() > 0 else None
            e0 = g.edge_attr.detach().clone() if getattr(g, "edge_attr",
                                                         None) is not None and g.edge_attr.numel() > 0 else None
            
            # 标准化
            if (mean_x is not None) and x0 is not None:
                g.x = (x0.float() - mean_x) / std_x
            if (mean_e is not None) and e0 is not None:
                g.edge_attr = (e0.float() - mean_e) / std_e
            
            # —— 统一影子字段（所有样本都“有且同维度”）——
            g.x_raw = x0 if x0 is not None else torch.empty((0, mean_x.numel()),
                                                            dtype=torch.float32) if mean_x is not None else torch.empty(
                (0, 0))
            if De is not None:
                # 有训练期的边维度定义，所有样本都提供 edge_attr_raw
                if e0 is not None:
                    g.edge_attr_raw = e0
                else:
                    # 没有边：提供形状 (0, De) 的占位
                    g.edge_attr_raw = torch.empty((0, De), dtype=torch.float32)
            else:
                # 训练期没有边特征 → 所有样本都不设置该键，避免多键集
                if hasattr(g, "edge_attr_raw"):
                    delattr(g, "edge_attr_raw")
            
            return g
        
        self.pyg_ds.transform = _standardize_graph  # ← 关键一步（不会造成数据泄露）  # NEW
        
        # 将标准化参数落盘，便于复现实验 / 推理阶段载入  # NEW
        scaler_path = os.path.join(self.cfg.out_dir, "x_scaler.json")
        payload = {
            "mean_x": (mean_x.tolist() if mean_x is not None else None),
            "std_x": (std_x.tolist() if std_x is not None else None),
            "mean_e": (mean_e.tolist() if mean_e is not None else None),
            "std_e": (std_e.tolist() if std_e is not None else None),
        }
        with open(scaler_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.logger.info(f"[standardize] saved scaler to: {scaler_path}")
        
        # 5) 构造 PyG 子集
        train_base = Subset(self.pyg_ds, self.train_idx)
        val_base = Subset(self.pyg_ds, self.val_idx)
        test_base = Subset(self.pyg_ds, self.test_idx)
        
        self.pyg_train = _GraphTransformWrapper(train_base, _standardize_graph)
        self.pyg_val = _GraphTransformWrapper(val_base, _standardize_graph)
        self.pyg_test = _GraphTransformWrapper(test_base, _standardize_graph)
        
        # 6) 由 PyG 子集生成 FusionDataset（与你现有工具函数保持一致）
        self.ds_train, self.ds_val, self.ds_test = make_fusion_datasets(
            self.pyg_train, self.pyg_val, self.cfg, pyg_test=self.pyg_test, logger=self.logger
        )

        return self  # 方便链式调用

    def build_loaders(self):
        pin = str(self.device).startswith("cuda")
        self.loader_train = DataLoader(
            self.ds_train, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, collate_fn=fusion_collate, pin_memory=pin
        )
        self.loader_val = DataLoader(
            self.ds_val, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, collate_fn=fusion_collate, pin_memory=pin
        )
        self.loader_test = DataLoader(
            self.ds_test, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, collate_fn=fusion_collate, pin_memory=pin
        )
        return self.loader_train, self.loader_val, self.loader_test

    # -------- 小工具 --------
    def infer_dims(self) -> Tuple[int, Optional[int], int]:
        """
        从训练集样本推断 (node_feat_dim, edge_feat_dim, out_classes)
        """
        # FusionDataset 的样本是一个 dict，含 "graph"
        sample = self.ds_train[0]["graph"] if len(self.ds_train) > 0 else self.ds_val[0]["graph"]
        node_feat_dim = int(sample.x.size(1)) if (hasattr(sample, "x") and sample.x is not None and sample.x.numel() > 0) else 0
        edge_feat_dim = int(sample.edge_attr.size(1)) if (hasattr(sample, "edge_attr") and sample.edge_attr is not None and sample.edge_attr.numel() > 0) else None
        out_classes   = len(getattr(self.pyg_ds, "class_to_idx", {})) or self.cfg.rf_num_classes
        return node_feat_dim, edge_feat_dim, out_classes

    def make_train_eval_loader(self):
        """
        复用你原脚本中“训练集评估版本”的构造；需要严格复刻时也可以做成独立 cfg 开关。
        """
        pin = str(self.device).startswith("cuda")
        ds_train_eval = FusionDataset(
            self.pyg_train,
            image_size=self.cfg.image_size,
            feat_stride=self.cfg.feat_stride,
            use_imagenet_norm=True,
            rf_csv=self.cfg.rf_csv,
            num_classes=self.cfg.rf_num_classes,
            rf_key_col="specimen_id",
            rf_pick_mode="logit_mean",
            rf_random_seed=self.cfg.split_seed,
            rf_verbose=False,
            morph_area_idx=0,
            morph_perim_idx=1,
            radius_alpha=1.0,
            min_radius_px=2.0,
            max_radius_px=32.0,
            pos_format=self.cfg.pos_format,
            verbose=False,
        )
        return DataLoader(
            ds_train_eval, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, collate_fn=fusion_collate, pin_memory=pin
        )
