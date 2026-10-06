# -*- coding: utf-8 -*-
"""
train_tri_modal_swin_fusion_v3_2.py

在 v3_1 基础上升级：✅ Region mask 不再只用最后一层(7×7) —— 改为：
- mask 在高分辨率 stage0(56×56) 生成（更细）
- 下采样到 56/28/14/7 多尺度做加权池化，融合得到 z_R（更稳更强）
- Swin 仍然只 forward 一次（features_only=True 一次拿到 4 个 stage）

其余：pretrained(离线 ckpt) / drop_path / freeze-unfreeze / 深监督 / aux退火 / 三对 contrast / EMA / 分层LR / AMPAPI 等保持。
"""

import os
import re
import math
import time
import json
import random
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, global_mean_pool

# ==== 外部 / 项目工具 ====
from FusionDataset import logger
from util.fusion_debug import guard_finite, DEBUG_NAN
from util.fusion_utils import (
    select_device, accuracy_top1, _fmt_eta,
)

from create_graph.config import SAVEROOT, IMAGEROOT
from util.data_module import WoodFusionDataModule

# Swin 来自 timm
import timm
import matplotlib.pyplot as plt
import csv


# =========================== Config ===========================

@dataclass
class TrainConfig:
    # 数据
    image_size: tuple = (224, 224)
    pos_format: str = "xy"  # 你的数据是 xy；内部会转成 yx 使用
    split_seed: int = 2025
    test_file: str = 'inputs/splits/val.txt'
    swin_ckpt_path: str = 'external_weights/swin_base/pytorch_model.bin'

    # ==== RF 概率（为 DataModule 兼容而保留）====
    rf_csv: str = 'inputs/ms_probabilities.csv'
    rf_num_classes: int = 7
    rf_force_off_end: int = 0
    rf_drop_end: int = 0
    rf_drop_value: float = 0.0

    # ==== 图像 / 特征网格步长 ====
    feat_stride: int = 4  # 224 -> 56 grid

    # 训练
    epochs: int = 300
    batch_size: int = 32
    num_workers: int = 3
    lr_head: float = 1e-4
    lr_backbone: float = 3e-5
    weight_decay: float = 1e-4
    amp: bool = False
    label_smoothing: float = 0.1
    grad_clip_norm: float = 1.0

    # Graph AE
    g_hidden: int = 128
    g_layers: int = 4
    g_dropout: float = 0.3

    # Swin
    swin_name: str = "swin_base_patch4_window7_224"
    d_model: int = 256
    swin_pretrained: bool = True
    swin_drop_path: float = 0.2

    # 区域 mask
    region_min_sigma: float = 1.0

    # ===== Region: 高分辨率 / 多尺度配置 =====
    region_use_multiscale: bool = True          # True: 56/28/14/7 多尺度融合；False: 只用 region_use_stage
    region_use_stage: int = 0                   # 单尺度时用哪个 stage（0=56,1=28,2=14,3=7）
    region_stage_indices: tuple = (0, 1, 2, 3)  # 多尺度时使用哪些 stage
    region_base_stage: int = 0                  # mask 在哪个 stage 分辨率上生成（推荐 0=56）
    region_recon_stage: int = 0                 # region_rec 用哪个 stage 的 weighted feature（推荐 0=56）

    # Fusion Transformer
    fusion_layers: int = 2
    fusion_heads: int = 4
    fusion_ffn_mult: int = 4

    # Loss 权重（会被 aux_factor 退火）
    w_graph_recon: float = 0.1
    w_img_recon: float = 1.0
    w_region_recon: float = 1.0

    # 深监督（每个模态一个辅助头）
    w_aux_cls: float = 0.2

    # Contrast（三对齐更稳）
    w_contrast: float = 0.3
    contrast_temperature: float = 0.2

    # Graph recon 归一化尺度（避免 g_recon 爆大）
    r_scale: float = 32.0
    pos_eps: float = 1e-6

    # 冻结/解冻 backbone
    freeze_swin_epochs: int = 5

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9997

    # 辅助损失退火：warmup -> cosine decay -> aux_min
    aux_warmup_epochs: int = 10
    aux_decay_epochs: int = 100
    aux_min: float = 0.1

    # 类别权重（不均衡时很关键）
    use_class_weight: bool = True

    # 日志 & 保存
    out_dir: str = 'outputs/ckeswin'
    save_best_name: str = "best.pt"
    save_last_name: str = "last.pt"
    log_every_n_steps: int = 20


# =========================== 常用工具 ===========================

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def contrastive_nt_xent(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    B = a.size(0)
    if B <= 1:
        return a.new_zeros(())
    sim = a @ b.t()
    logits_ab = sim / temperature
    logits_ba = sim.t() / temperature
    target = torch.arange(B, device=a.device, dtype=torch.long)
    loss_ab = F.cross_entropy(logits_ab, target)
    loss_ba = F.cross_entropy(logits_ba, target)
    return 0.5 * (loss_ab + loss_ba)


def _safe_shape(t):
    return tuple(t.shape) if (t is not None and hasattr(t, "shape")) else "None"


def _grid_hw(cfg: TrainConfig):
    H, W = cfg.image_size
    return H // cfg.feat_stride, W // cfg.feat_stride


def _to_yx(pos: torch.Tensor, pos_format: str):
    """输入可能是 xy 或 yx；统一输出 yx"""
    if pos is None:
        return None
    if pos_format.lower() == "xy":
        return pos[:, [1, 0]]
    return pos


def _maybe_to_grid_space(yx: torch.Tensor, cfg: TrainConfig):
    """自动判别 node_pos_feat 的尺度：pixel->grid 或已是 grid"""
    if yx is None or yx.numel() == 0:
        return yx
    H_img, W_img = cfg.image_size
    gH, gW = _grid_hw(cfg)
    maxv = float(yx.max().item())
    if maxv > max(gH, gW) + 5 and maxv <= max(H_img, W_img) + 5:
        yx = yx / float(cfg.feat_stride)
    return yx


def _maybe_r_to_grid(r: torch.Tensor, cfg: TrainConfig):
    """自动判别 r 的尺度：pixel 半径 -> grid 半径"""
    if r is None or r.numel() == 0:
        return r
    H_img, W_img = cfg.image_size
    maxv = float(r.max().item())
    if maxv > cfg.r_scale + 10 and maxv <= max(H_img, W_img) + 10:
        r = r / float(cfg.feat_stride)
    return r


def _scale_grid_to_feat(yx_grid: torch.Tensor, r_grid: torch.Tensor, gH: int, gW: int, Hf: int, Wf: int):
    """把 56×56 grid 坐标缩放到任意特征图 Hf×Wf"""
    if yx_grid is None or yx_grid.numel() == 0:
        return yx_grid, r_grid

    yx = yx_grid.clone()
    scale_y = (Hf - 1) / max(1.0, float(gH - 1))
    scale_x = (Wf - 1) / max(1.0, float(gW - 1))
    yx[:, 0] = yx[:, 0] * scale_y
    yx[:, 1] = yx[:, 1] * scale_x

    if r_grid is not None and r_grid.numel() > 0:
        r = r_grid.clone()
        r = r * (0.5 * (scale_y + scale_x))
    else:
        r = r_grid
    return yx, r


def aux_factor(epoch: int, cfg: TrainConfig) -> float:
    """辅助损失乘子：warmup 到 1，然后余弦衰减到 cfg.aux_min"""
    if cfg.aux_decay_epochs <= 0:
        return 1.0
    if epoch <= cfg.aux_warmup_epochs:
        return max(0.0, float(epoch) / max(1, cfg.aux_warmup_epochs))
    t = min(1.0, float(epoch - cfg.aux_warmup_epochs) / max(1, cfg.aux_decay_epochs))
    cosv = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
    return cfg.aux_min + (1.0 - cfg.aux_min) * cosv


class ModelEMA:
    """EMA 权重：验证/保存 best 用 EMA 往往更高（兼容 Long/Bool buffers）"""
    def __init__(self, model: nn.Module, decay: float = 0.9997):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in msd.items():
            v = v.detach()
            if k not in self.shadow:
                self.shadow[k] = v.clone()
                continue

            sv = self.shadow[k]
            if torch.is_floating_point(v) and torch.is_floating_point(sv) and (sv.dtype == v.dtype):
                sv.mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.clone()

    @contextmanager
    def apply(self, model: nn.Module):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(self._backup, strict=False)
            self._backup = None


def compute_class_weights_fusion_robust(
    num_classes: int,
    ds=None,
    loader=None,
    class_to_idx: dict = None,
    logger=None,
    max_samples: int = 0,
    clamp_max: float = 10.0,
    eps: float = 1e-12,
):
    """
    适配你的 FusionDataset:
      sample 是 dict: {'graph': Data, 'image':..., 'meta':...}

    取标签优先级：
      1) sample['graph'].y / sample.y / obj.y
      2) sample['meta'] 中可能存在的 label
      3) 从 img_path 路径组件匹配 class_to_idx 的类别文件夹名
      4) 如果 dataset 统计不到（n==0），用 loader 的 batch_graph.y 兜底
    """

    def _to_int_label(y):
        if y is None:
            return None
        if torch.is_tensor(y):
            if y.numel() == 0:
                return None
            yv = y.detach().view(-1)
            if y.ndim == 1 and y.numel() == num_classes and (yv.min() >= 0) and (yv.max() <= 1.0 + 1e-3):
                return int(torch.argmax(yv).item())
            return int(yv[0].item())
        if isinstance(y, np.ndarray):
            if y.size == 0:
                return None
            if y.ndim == 1 and y.size == num_classes:
                return int(np.argmax(y))
            return int(np.reshape(y, (-1,))[0])
        if isinstance(y, (list, tuple)):
            if len(y) == 0:
                return None
            if len(y) == num_classes and all(isinstance(v, (int, float, np.number)) for v in y):
                return int(np.argmax(np.array(y, dtype=float)))
            return int(y[0])
        if isinstance(y, (int, np.integer)):
            return int(y)
        if isinstance(y, (float, np.floating)):
            return int(y)
        if isinstance(y, str) and y.strip().isdigit():
            return int(y.strip())
        return None

    def _extract_from_obj(obj):
        if obj is None:
            return None, None, None

        if isinstance(obj, dict):
            if "graph" in obj:
                yi, ip, meta = _extract_from_obj(obj["graph"])
                ip = ip or obj.get("img_path", None) or obj.get("meta", {}).get("img_path", None)
                meta = meta or obj.get("meta", None)
                if yi is not None:
                    return yi, ip, meta
            for k in ("y", "label", "target", "class_idx"):
                if k in obj:
                    yi = _to_int_label(obj[k])
                    if yi is not None:
                        return yi, obj.get("img_path", None), obj.get("meta", None)
            return None, obj.get("img_path", None) or obj.get("meta", {}).get("img_path", None), obj.get("meta", None)

        if isinstance(obj, (tuple, list)):
            ip = None
            meta = None
            for it in obj:
                yi, ip2, meta2 = _extract_from_obj(it)
                ip = ip or ip2
                meta = meta or meta2
                if yi is not None:
                    return yi, ip, meta
            return None, ip, meta

        if hasattr(obj, "y"):
            yi = _to_int_label(getattr(obj, "y", None))
            if yi is not None:
                return yi, getattr(obj, "img_path", None), getattr(obj, "meta", None)

        ip = getattr(obj, "img_path", None)
        meta = getattr(obj, "meta", None)
        return None, ip, meta

    def _label_from_meta(meta):
        if not isinstance(meta, dict):
            return None
        for k in ("y", "label", "target", "class_idx"):
            if k in meta:
                yi = _to_int_label(meta[k])
                if yi is not None:
                    return yi
        if class_to_idx is not None:
            for k in ("class_name", "species", "category", "label_name"):
                if k in meta:
                    name = str(meta[k]).strip()
                    if name in class_to_idx:
                        return int(class_to_idx[name])
        return None

    def _label_from_path(img_path):
        if img_path is None or class_to_idx is None:
            return None
        p = str(img_path).replace("\\", "/").lower()
        parts = re.split(r"/+", p)
        cmap = {str(k).lower(): int(v) for k, v in class_to_idx.items()}
        for part in parts[::-1]:
            if part in cmap:
                return cmap[part]
        for part in parts[::-1]:
            for k, v in cmap.items():
                if k in part:
                    return v
        return None

    def _acc(counts, yi):
        if yi is None:
            return False
        try:
            yi = int(yi)
        except Exception:
            return False
        if 0 <= yi < num_classes:
            counts[yi] += 1
            return True
        return False

    def _make_weight(counts):
        total = counts.sum().clamp_min(1.0)
        w = total / (counts.clamp_min(1.0) * float(num_classes))
        w = w / (w.mean().clamp_min(eps))
        if clamp_max is not None and clamp_max > 0:
            w = w.clamp(min=0.0, max=float(clamp_max))
        return w

    counts = torch.zeros(num_classes, dtype=torch.float64)
    n = 0

    if ds is not None:
        L = len(ds)
        if max_samples and 0 < max_samples < L:
            idxs = np.linspace(0, L - 1, max_samples, dtype=int).tolist()
        else:
            idxs = range(L)

        for i in idxs:
            sample = ds[i]
            yi, ip, meta = _extract_from_obj(sample)
            if yi is None:
                yi = _label_from_meta(meta)
            if yi is None:
                yi = _label_from_path(ip)
            if _acc(counts, yi):
                n += 1

    counts = counts.float()
    info = {"source": "dataset", "n_effective": int(n)}

    if (n <= 0 or counts.sum().item() <= 0) and loader is not None:
        counts2 = torch.zeros(num_classes, dtype=torch.float64)
        n2 = 0
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                batch_graph = batch[0]
            else:
                batch_graph = batch

            if batch_graph is None or not hasattr(batch_graph, "y"):
                continue
            y = batch_graph.y
            if y is None or (not torch.is_tensor(y)) or y.numel() == 0:
                continue
            yv = y.view(-1).detach().cpu()
            for v in yv.tolist():
                if _acc(counts2, int(v)):
                    n2 += 1
        counts = counts2.float()
        n = n2
        info = {"source": "loader", "n_effective": int(n)}

    if n <= 0 or counts.sum().item() <= 0:
        w = torch.ones(num_classes, dtype=torch.float32)
        info["warning"] = "No labels found -> return uniform weights."
        return w, counts.float(), int(n), info

    w = _make_weight(counts)

    if (not torch.isfinite(w).all()) or (w.max().item() <= 0):
        w = torch.ones(num_classes, dtype=torch.float32)
        info["warning"] = "Invalid weights -> return uniform weights."

    return w.float(), counts.float(), int(n), info


# ===================== Graph AutoEncoder ======================

class GraphEncoder(nn.Module):
    """简化版 DeepGINE：多层 GINEConv + LN + 残差 + global_mean_pool"""
    def __init__(self, in_channels, edge_dim, hidden, num_layers, dropout=0.3):
        super().__init__()
        self.hidden = hidden
        self.dropout = float(dropout)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        mlp_in = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.layers.append(GINEConv(mlp_in, edge_dim=edge_dim if edge_dim is not None else 0))
        self.norms.append(nn.LayerNorm(hidden))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
            )
            self.layers.append(GINEConv(mlp, edge_dim=edge_dim if edge_dim is not None else 0))
            self.norms.append(nn.LayerNorm(hidden))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch_vec = data.batch

        h = x
        for conv, norm in zip(self.layers, self.norms):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if h_in.shape == h.shape:
                h = h + h_in

        node_emb = h
        graph_emb = global_mean_pool(node_emb, batch_vec)
        return node_emb, graph_emb


class GraphDecoder(nn.Module):
    """重建 node-level：x / pos(yx) / r"""
    def __init__(self, hidden, node_in_dim, pos_dim=2, r_dim=1):
        super().__init__()
        self.dec_x = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, node_in_dim),
        )
        self.dec_pos = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, pos_dim),
        )
        self.dec_r = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, r_dim),
        )

    def forward(self, node_emb):
        x_hat = self.dec_x(node_emb)
        pos_hat = self.dec_pos(node_emb)
        r_hat = self.dec_r(node_emb)
        return x_hat, pos_hat, r_hat


# ===================== Swin Encoder (Multi-Scale) =====================
class SwinEncoder(nn.Module):
    """
    ✅ 一次 forward 得到多尺度特征:
      feats[0]=56x56, feats[1]=28x28, feats[2]=14x14, feats[3]=7x7 (对 224 输入)

    ✅ 离线 ckpt 自动适配 key：
      - 常见差异：layers.0.xxx  vs  layers_0.xxx
      - 自动选择匹配率最高的映射策略
      - 过滤掉模型里不存在的 key（例如 relative_position_index / attn_mask 等 buffer）
    """
    def __init__(self, model_name: str, d_model: int, pretrained: bool = True, drop_path: float = 0.2,
                 ckpt_path: str = ""):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,          # 离线
            features_only=True,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path,
        )

        # ===== 找到最可能的“真正底座模型”来 load 权重 =====
        def _collect_candidates(m):
            cands = []
            seen = set()

            def add(x):
                if x is None:
                    return
                if id(x) in seen:
                    return
                seen.add(id(x))
                cands.append(x)

            add(m)
            for attr in ("model", "net", "backbone"):
                add(getattr(m, attr, None))
                add(getattr(getattr(m, attr, None), "model", None))
            return cands

        candidates = _collect_candidates(self.backbone)

        if not ckpt_path:
            raise RuntimeError("[Swin] ckpt_path is empty in offline mode; would be random init.")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[Swin] ckpt not found: {ckpt_path}")

        def _extract_state_dict(obj):
            if isinstance(obj, dict) and "state_dict" in obj:
                return obj["state_dict"]
            if isinstance(obj, dict) and "model" in obj:
                return obj["model"]
            return obj

        def _strip_prefix(k: str):
            for p in ("module.", "model.", "backbone."):
                if k.startswith(p):
                    k = k[len(p):]
            return k

        sd = torch.load(ckpt_path, map_location="cpu")
        sd = _extract_state_dict(sd)
        if not isinstance(sd, dict):
            raise RuntimeError("[Swin] loaded checkpoint is not a state_dict-like dict.")
        sd = {_strip_prefix(k): v for k, v in sd.items()}

        # ===== key remap 策略 =====
        def _remap_layers_dot_to_us(k: str) -> str:
            # layers.0.xxx -> layers_0.xxx
            return re.sub(r"^layers\.(\d+)\.", r"layers_\1.", k)

        def _remap_layers_us_to_dot(k: str) -> str:
            # layers_0.xxx -> layers.0.xxx
            return re.sub(r"^layers_(\d+)\.", r"layers.\1.", k)

        def _apply_remap(sd_in: dict, fn):
            out = {}
            remapped = 0
            collisions = 0
            for k, v in sd_in.items():
                k2 = fn(k)
                if k2 != k:
                    remapped += 1
                if k2 in out:
                    collisions += 1
                    continue
                out[k2] = v
            return out, {"remapped": remapped, "collisions": collisions}

        def _match_ratio(sd_in: dict, model_keys: set) -> float:
            inter = len(model_keys.intersection(sd_in.keys()))
            return inter / max(1, len(model_keys))

        # ===== 选择 “候选模型 + key映射策略” 里匹配率最高的组合 =====
        best = None
        best_detail = None

        for m in candidates:
            model_keys = set(m.state_dict().keys())

            sd_raw = sd
            r0 = _match_ratio(sd_raw, model_keys)

            sd_d2u, meta_d2u = _apply_remap(sd_raw, _remap_layers_dot_to_us)
            r1 = _match_ratio(sd_d2u, model_keys)

            sd_u2d, meta_u2d = _apply_remap(sd_raw, _remap_layers_us_to_dot)
            r2 = _match_ratio(sd_u2d, model_keys)

            # 选这个 model 下最好的策略
            ratios = [
                ("raw",  sd_raw, r0, {"remapped": 0, "collisions": 0}),
                ("d2u",  sd_d2u, r1, meta_d2u),
                ("u2d",  sd_u2d, r2, meta_u2d),
            ]
            name, sd_best, r_best, meta_best = max(ratios, key=lambda x: x[2])

            if best is None or r_best > best:
                best = r_best
                best_detail = (m, model_keys, name, sd_best, meta_best)

        target_model, target_keys, strategy, sd_best, meta = best_detail
        bm = getattr(target_model, "model", None) or getattr(target_model, "net", None) or getattr(target_model, "backbone", None)
        self.base_model = bm if bm is not None else target_model

        logger.info(f"[Swin] choose target={type(target_model).__name__} strategy={strategy} "
                    f"ratio={best:.3f} meta={meta}")

        # ===== 过滤掉模型里不存在的 key，避免 unexpected 巨多 =====
        sd_best = {k: v for k, v in sd_best.items() if k in target_keys}

        missing, unexpected = target_model.load_state_dict(sd_best, strict=False)
        logger.info(f"[Swin] loaded ckpt={ckpt_path} | loaded_ratio={best:.3f} "
                    f"| missing={len(missing)} unexpected={len(unexpected)}")

        if best < 0.90:
            some_sd = list(sd_best.keys())[:20]
            some_mk = list(target_keys)[:20]
            raise RuntimeError(
                f"[Swin] state_dict key mismatch still too large (loaded_ratio={best:.3f}).\n"
                f"  sample ckpt keys: {some_sd}\n"
                f"  sample model keys: {some_mk}\n"
                f"  -> check model_name/ckpt/timm version."
            )

        # stage 通道数
        if hasattr(self.backbone, "feature_info"):
            self.stage_channels = list(self.backbone.feature_info.channels())
        else:
            self.stage_channels = []

        # out_ch = 最后一层通道
        if self.stage_channels and len(self.stage_channels) >= 4:
            self.out_ch = int(self.stage_channels[-1])
        else:
            # forward 再兜底修正
            self.out_ch = 1024

        self.proj = nn.Linear(self.out_ch, d_model)

    def forward(self, x):
        feats = self.backbone(x)  # list of feature maps

        if not isinstance(feats, (list, tuple)) or len(feats) == 0:
            raise RuntimeError("[Swin] features_only backbone returned empty.")

        # === 关键修复：如果是 NHWC -> 转成 NCHW ===
        feats_fixed = []
        for f in feats:
            if (f.dim() == 4
                and f.shape[1] in (7, 14, 28, 56)   # H
                and f.shape[2] in (7, 14, 28, 56)   # W
                and f.shape[3] >= 64):              # C 很大（128/256/512/1024）
                f = f.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
            feats_fixed.append(f)
        feats = feats_fixed

        # 兜底补全 stage_channels/out_ch/proj
        if (not getattr(self, "stage_channels", None)) or (len(self.stage_channels) != len(feats)):
            self.stage_channels = [int(f.shape[1]) for f in feats]  # NCHW: C 在 dim=1
            self.out_ch = int(self.stage_channels[-1])
            if self.proj.in_features != self.out_ch:
                self.proj = nn.Linear(self.out_ch, self.proj.out_features).to(feats[-1].device)

        feat_last = feats[-1]                        # [B,C,7,7]
        pooled = feat_last.mean(dim=(2, 3))          # [B,C]
        z = self.proj(pooled)                        # [B,d_model]
        return z, feat_last, feats



class SimpleImageDecoder(nn.Module):
    """简单上采样 decoder，把 feature map 重建成输入图像大小"""
    def __init__(self, in_ch: int, out_ch: int = 3, num_upsample: int = 3):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_upsample):
            layers.extend([
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ch, max(8, ch // 2), kernel_size=3, padding=1),
                nn.BatchNorm2d(max(8, ch // 2)),
                nn.ReLU(inplace=True),
            ])
            ch = max(8, ch // 2)
        layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat, out_size):
        x = self.net(feat)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


# ===================== Region Mask (Gaussian) =====================

def gaussian_weight_map(Hf, Wf, yx, r, device, min_sigma=1.0, sigma_scale=1.0):
    """
    yx: [Ni, 2], r: [Ni,1], 均在 feature map 索引坐标系（0..Hf-1, 0..Wf-1）
    输出: [Hf, Wf] 归一化权重图（sum=1）
    """
    if yx is None or yx.numel() == 0:
        return torch.full((Hf, Wf), 1.0 / (Hf * Wf), device=device)

    W = torch.zeros(Hf, Wf, device=device)
    for j in range(yx.size(0)):
        y0, x0 = yx[j]
        sigma = torch.clamp(r[j, 0] * sigma_scale, min=min_sigma)
        R = int(2.5 * sigma.item() + 1)
        dy = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
        dx = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
        dyy, dxx = torch.meshgrid(dy, dx, indexing="ij")
        G = torch.exp(-(dyy ** 2 + dxx ** 2) / (2 * sigma ** 2 + 1e-6))
        yy = torch.clamp(y0 + dyy, 0, Hf - 1).long()
        xx = torch.clamp(x0 + dxx, 0, Wf - 1).long()
        W.index_put_((yy, xx), G, accumulate=True)

    s = W.sum()
    if s > 0:
        W = W / s
    else:
        W = torch.full((Hf, Wf), 1.0 / (Hf * Wf), device=device)
    return W


# ===================== Fusion Transformer =====================

class TriModalFusionTransformer(nn.Module):
    """输入三个 token: z_G, z_I, z_R + [CLS] -> 分类"""
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2, ffn_mult: int = 4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ffn_mult,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, zG, zI, zR):
        B = zG.size(0)
        tokens = torch.stack([zG, zI, zR], dim=1)  # [B, 3, d_model]
        cls = self.cls_token.expand(B, -1, -1)     # [B, 1, d_model]
        x = torch.cat([cls, tokens], dim=1)        # [B, 4, d_model]
        x = self.encoder(x)
        return x[:, 0, :]                          # CLS


# ===================== Tri-Modal Swin Fusion Model =====================

class TriModalSwinFusionModel(nn.Module):
    """
    1) Graph AE -> z_G
    2) Swin Global -> z_I（来自最后层 pooled）
    3) Region：mask 在高分辨率 stage 生成；(可选) 多尺度池化融合 -> z_R
    4) Transformer 融合 -> 分类
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, out_classes: int, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model

        # Graph AE
        self.graph_enc = GraphEncoder(
            in_channels=node_feat_dim,
            edge_dim=edge_feat_dim,
            hidden=cfg.g_hidden,
            num_layers=cfg.g_layers,
            dropout=cfg.g_dropout,
        )
        self.graph_dec = GraphDecoder(
            hidden=cfg.g_hidden,
            node_in_dim=node_feat_dim,
            pos_dim=2,
            r_dim=1,
        )
        self.graph_proj = nn.Linear(cfg.g_hidden, d_model)

        # Swin（共享）
        self.swin_enc = SwinEncoder(
            model_name=cfg.swin_name,
            d_model=d_model,
            pretrained=False,                 # 离线 ckpt
            drop_path=cfg.swin_drop_path,
            ckpt_path=cfg.swin_ckpt_path,
        )

        # 最后一层通道（用于 global decoder）
        in_ch_last = self.swin_enc.out_ch

        # Global 图像重建（仍保留，但会退火到 0）
        self.img_dec_global = SimpleImageDecoder(in_ch=in_ch_last, out_ch=3, num_upsample=3)

        # ===== Region: 多尺度/高分辨率 pooling =====
        self.region_use_multiscale = bool(cfg.region_use_multiscale)
        self.region_base_stage = int(cfg.region_base_stage)
        self.region_recon_stage = int(cfg.region_recon_stage)

        stage_chs = getattr(self.swin_enc, "stage_channels", None)
        if not stage_chs or len(stage_chs) < 4:
            # 兜底：不太会发生
            stage_chs = [in_ch_last // 8, in_ch_last // 4, in_ch_last // 2, in_ch_last]
        self.stage_chs = stage_chs

        if self.region_use_multiscale:
            self.region_stage_indices = list(cfg.region_stage_indices)
            self.region_projs = nn.ModuleList([
                nn.Linear(self.stage_chs[i], d_model) for i in self.region_stage_indices
            ])
            self.region_stage_logits = nn.Parameter(torch.zeros(len(self.region_stage_indices)))
        else:
            self.region_use_stage = int(cfg.region_use_stage)
            self.region_proj_single = nn.Linear(self.stage_chs[self.region_use_stage], d_model)

        # Region 重建：用指定 recon stage 的 weighted feature
        recon_ch = self.stage_chs[self.region_recon_stage]
        self.img_dec_region = SimpleImageDecoder(in_ch=recon_ch, out_ch=3, num_upsample=3)

        # Fusion
        self.fusion = TriModalFusionTransformer(
            d_model=d_model,
            nhead=cfg.fusion_heads,
            num_layers=cfg.fusion_layers,
            ffn_mult=cfg.fusion_ffn_mult,
        )

        # 主分类头
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, out_classes),
        )

        # 深监督：三个模态各一个辅助头
        self.aux_head_G = nn.Linear(d_model, out_classes)
        self.aux_head_I = nn.Linear(d_model, out_classes)
        self.aux_head_R = nn.Linear(d_model, out_classes)

        # region sigma scale 参数
        init = math.log(math.e - 1.0)
        self.sigma_scale_param = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    @staticmethod
    def _per_graph_split(node_tensor: torch.Tensor, batch_vec: torch.Tensor, B: int):
        return [node_tensor[batch_vec == i] for i in range(B)]

    def forward(self, batch_graph, batch_img, return_details: bool = False):
        device = batch_img.device

        # ===== Graph AE =====
        node_emb, graph_emb = self.graph_enc(batch_graph)
        z_G = self.graph_proj(graph_emb)
        x_hat, pos_hat, r_hat = self.graph_dec(node_emb)

        # ===== Swin (multi-scale once) =====
        z_I, feat_last, feats_stages = self.swin_enc(batch_img)  # feat_last: stage3(7x7)
        img_rec = self.img_dec_global(feat_last, out_size=batch_img.shape[-2:])

        # ===== Region (hi-res mask + multi-scale pooling) =====
        yx_all = getattr(batch_graph, "node_pos_feat", None)
        r_all  = getattr(batch_graph, "node_r_feat", None)

        if yx_all is None:
            yx_all = torch.zeros(node_emb.size(0), 2, device=device)
        else:
            yx_all = yx_all.to(device).float()
            yx_all = _to_yx(yx_all, self.cfg.pos_format)
            yx_all = _maybe_to_grid_space(yx_all, self.cfg)   # -> 56 grid

        if r_all is None:
            r_all = torch.ones(node_emb.size(0), 1, device=device)
        else:
            r_all = r_all.to(device).float()
            r_all = _maybe_r_to_grid(r_all, self.cfg)         # -> 56 grid radius

        # --- base stage（推荐 stage0=56x56）生成 mask ---
        feat_base = feats_stages[self.region_base_stage]       # [B,Cb,Hb,Wb]
        Hb, Wb = feat_base.shape[-2], feat_base.shape[-1]

        gH, gW = _grid_hw(self.cfg)                            # 通常 56,56
        yx_all_base, r_all_base = _scale_grid_to_feat(yx_all, r_all, gH, gW, Hb, Wb)

        batch_vec = batch_graph.batch
        B = batch_img.size(0)
        yx_lists = self._per_graph_split(yx_all_base, batch_vec, B)
        r_lists  = self._per_graph_split(r_all_base,  batch_vec, B)

        sigma_scale = F.softplus(self.sigma_scale_param) + 1e-3

        mask_base_list = []
        mask_imgs_list = []

        for i in range(B):
            m = gaussian_weight_map(
                Hb, Wb, yx_lists[i], r_lists[i],
                device=device,
                min_sigma=self.cfg.region_min_sigma,
                sigma_scale=sigma_scale,
            )  # [Hb,Wb], sum=1
            mask_base_list.append(m.unsqueeze(0))  # [1,Hb,Wb]

            # 用于 region_target：上采样到原图并归一到 0..1（按 max）
            mi = F.interpolate(
                m.unsqueeze(0).unsqueeze(0),
                size=batch_img.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)
            mmax = mi.max().clamp_min(1e-6)
            mask_imgs_list.append((mi / mmax).clamp(0.0, 1.0).unsqueeze(0))  # [1,H,W]

        mask_base = torch.stack(mask_base_list, dim=0)  # [B,1,Hb,Wb]
        mask_imgs = torch.stack(mask_imgs_list, dim=0)  # [B,1,H,W]

        def _resize_mask_sum1(mask_b1hw, out_hw):
            m = F.interpolate(mask_b1hw, size=out_hw, mode="bilinear", align_corners=False)
            s = m.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            return m / s

        # (A) region pooling -> z_R
        if self.region_use_multiscale:
            z_list = []
            for k, sidx in enumerate(self.region_stage_indices):
                feat_s = feats_stages[sidx]                    # [B,Cs,Hs,Ws]
                Hs, Ws = feat_s.shape[-2], feat_s.shape[-1]
                mask_s = _resize_mask_sum1(mask_base, (Hs, Ws))  # [B,1,Hs,Ws], sum=1

                region_vec = (feat_s * mask_s).sum(dim=(2, 3))   # [B,Cs]
                z_s = self.region_projs[k](region_vec)           # [B,d_model]
                z_list.append(z_s)

            w = F.softmax(self.region_stage_logits, dim=0)       # [K]
            z_R = torch.zeros_like(z_list[0])
            for wi, zi in zip(w, z_list):
                z_R = z_R + wi * zi
        else:
            sidx = self.region_use_stage
            feat_s = feats_stages[sidx]
            Hs, Ws = feat_s.shape[-2], feat_s.shape[-1]
            mask_s = _resize_mask_sum1(mask_base, (Hs, Ws))
            region_vec = (feat_s * mask_s).sum(dim=(2, 3))
            z_R = self.region_proj_single(region_vec)

        # (B) region reconstruction：用指定 recon stage 的 weighted feature
        recon_feat = feats_stages[self.region_recon_stage]       # [B,Cr,Hr,Wr]
        Hr, Wr = recon_feat.shape[-2], recon_feat.shape[-1]
        mask_rec = _resize_mask_sum1(mask_base, (Hr, Wr))        # [B,1,Hr,Wr]

        feat_weighted_rec = recon_feat * mask_rec
        region_target = batch_img * mask_imgs
        region_rec = self.img_dec_region(feat_weighted_rec, out_size=batch_img.shape[-2:])

        # ===== Fusion + 主分类 =====
        z_fused = self.fusion(z_G, z_I, z_R)
        logits = self.cls_head(z_fused)

        if not return_details:
            return logits

        return logits, {
            "z_G": z_G,
            "z_I": z_I,
            "z_R": z_R,
            "graph_recon": {"x_hat": x_hat, "pos_hat": pos_hat, "r_hat": r_hat},
            "img_rec": img_rec,
            "region_imgs": region_target,
            "region_rec": region_rec,
            "debug": {
                "stages": [tuple(f.shape) for f in feats_stages],
                "base_stage": int(self.region_base_stage),
                "recon_stage": int(self.region_recon_stage),
                "HbWb": (int(Hb), int(Wb)),
            }
        }


# =========================== History & Plot ===========================

def append_history(history, epoch, split, loss, acc, lr):
    history.append({
        "epoch": int(epoch),
        "split": split,
        "loss": float(loss),
        "acc": float(acc),
        "lr": float(lr),
    })


def save_history_csv(path, history):
    if not history:
        return
    keys = ["epoch", "split", "loss", "acc", "lr"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in history:
            w.writerow(r)


def plot_history_curves(history, out_dir, prefix="history"):
    if not history:
        return
    os.makedirs(out_dir, exist_ok=True)
    epochs = sorted({r["epoch"] for r in history})

    def seq(split, key):
        m = {}
        for r in history:
            if r["split"] == split:
                m[r["epoch"]] = r[key]
        return [m.get(e, float("nan")) for e in epochs]

    plt.figure()
    plt.plot(epochs, seq("train", "loss"), label="train")
    plt.plot(epochs, seq("val", "loss"), label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs. Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_loss.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, seq("train", "acc"), label="train")
    plt.plot(epochs, seq("val", "acc"), label="val")
    plt.xlabel("epoch"); plt.ylabel("acc"); plt.title("Accuracy vs. Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_acc.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, seq("train", "lr"), label="lr")
    plt.xlabel("epoch"); plt.ylabel("lr"); plt.title("LR vs. Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_lr.png"), dpi=200)
    plt.close()


# =========================== Train / Eval ===========================

def run_one_epoch(model, loader, optimizer, scaler, device, criterion, cfg: TrainConfig,
                  step_base=0, train=True, epoch=1, ema=None, eta_state=None):
    phase = "train" if train else "val"
    model.train(train)

    device_type = "cuda" if (isinstance(device, torch.device) and device.type == "cuda") else "cpu"

    total_loss, total_acc, total_n = 0.0, 0.0, 0
    step = step_base
    step_time_ema = None
    shapes_logged = False

    gH, gW = _grid_hw(cfg)
    pos_scale = torch.tensor([max(1, gH - 1), max(1, gW - 1)], dtype=torch.float32, device=device)
    r_scale = float(cfg.r_scale)

    af = aux_factor(epoch, cfg)

    for bidx, (batch_graph, batch_img, batch_rf, meta) in enumerate(loader):
        if batch_graph is None:
            continue

        if not shapes_logged and hasattr(batch_graph, "x"):
            has_xraw = hasattr(batch_graph, "x_raw")
            xm, xs = batch_graph.x.mean().item(), batch_graph.x.std().item()
            logger.info(f"[{phase}] x mean={xm:.4f}, std={xs:.4f} | has_x_raw={has_xraw}")

        if not shapes_logged and hasattr(batch_graph, "edge_attr") and batch_graph.edge_attr is not None:
            em, es = batch_graph.edge_attr.mean().item(), batch_graph.edge_attr.std().item()
            logger.info(f"[{phase}] edge_attr mean={em:.4f}, std={es:.4f}")

        batch_img = batch_img.to(device, non_blocking=True)
        try:
            batch_graph = batch_graph.to(device)
        except Exception:
            pass

        y = batch_graph.y.view(-1).to(device)

        if not shapes_logged:
            logger.info(
                f"[{phase}] batch0 shapes | "
                f"img={_safe_shape(batch_img)} | "
                f"x={_safe_shape(getattr(batch_graph, 'x', None))} | "
                f"edge_attr={_safe_shape(getattr(batch_graph, 'edge_attr', None))} | "
                f"node_pos_feat={_safe_shape(getattr(batch_graph, 'node_pos_feat', None))} | "
                f"node_r_feat={_safe_shape(getattr(batch_graph, 'node_r_feat', None))}"
            )
            shapes_logged = True

        guard_finite("inputs/img", batch_img)
        guard_finite("inputs/graph_x", getattr(batch_graph, "x", None))
        guard_finite("inputs/graph_edge_attr", getattr(batch_graph, "edge_attr", None))
        guard_finite("inputs/node_pos_feat", getattr(batch_graph, "node_pos_feat", None))
        guard_finite("inputs/node_r_feat", getattr(batch_graph, "node_r_feat", None))

        t_step_start = time.perf_counter()

        with torch.amp.autocast(device_type=device_type, enabled=cfg.amp):
            logits, info = model(batch_graph, batch_img, return_details=True)
            guard_finite("logits", logits)

            # ===== 主分类 CE =====
            cls_loss = criterion(logits, y)

            # ===== 深监督（辅助分类）=====
            aux_logits_G = model.aux_head_G(info["z_G"])
            aux_logits_I = model.aux_head_I(info["z_I"])
            aux_logits_R = model.aux_head_R(info["z_R"])
            aux_cls_loss = (criterion(aux_logits_G, y) + criterion(aux_logits_I, y) + criterion(aux_logits_R, y)) / 3.0

            # ===== Graph reconstruction（做尺度归一）=====
            x_hat = info["graph_recon"]["x_hat"]
            pos_hat = info["graph_recon"]["pos_hat"]
            r_hat = info["graph_recon"]["r_hat"]

            x_loss = logits.new_zeros(())
            pos_loss = logits.new_zeros(())
            r_loss = logits.new_zeros(())

            if getattr(batch_graph, "x", None) is not None:
                x_loss = F.mse_loss(x_hat, batch_graph.x)

            if getattr(batch_graph, "node_pos_feat", None) is not None:
                pos_tgt = batch_graph.node_pos_feat.to(pos_hat.device).float()
                pos_tgt = _to_yx(pos_tgt, cfg.pos_format)
                pos_tgt = _maybe_to_grid_space(pos_tgt, cfg)
                pos_loss = F.mse_loss(
                    pos_hat / (pos_scale + cfg.pos_eps),
                    pos_tgt / (pos_scale + cfg.pos_eps),
                )

            if getattr(batch_graph, "node_r_feat", None) is not None:
                r_tgt = batch_graph.node_r_feat.to(r_hat.device).float()
                r_tgt = _maybe_r_to_grid(r_tgt, cfg)
                r_loss = F.mse_loss(r_hat / (r_scale + 1e-6), r_tgt / (r_scale + 1e-6))

            graph_loss = x_loss + pos_loss + r_loss

            # ===== Image reconstruction（global）=====
            img_loss = F.mse_loss(info["img_rec"], batch_img)

            # ===== Region reconstruction（target = img * mask）=====
            region_loss = F.mse_loss(info["region_rec"], info["region_imgs"])

            # ===== Contrast：三对齐更稳 =====
            z_Gn = F.normalize(info["z_G"], dim=1)
            z_In = F.normalize(info["z_I"], dim=1)
            z_Rn = F.normalize(info["z_R"], dim=1)
            t = cfg.contrast_temperature
            c1 = contrastive_nt_xent(z_Gn, z_Rn, temperature=t)
            c2 = contrastive_nt_xent(z_Gn, z_In, temperature=t)
            c3 = contrastive_nt_xent(z_In, z_Rn, temperature=t)
            contrast_loss = (c1 + c2 + c3) / 3.0

            # ===== 总 loss：辅助任务自动退火，后期让分类主导 =====
            loss = (
                cls_loss
                + cfg.w_aux_cls * aux_cls_loss
                + af * (cfg.w_graph_recon * graph_loss + cfg.w_img_recon * img_loss + cfg.w_region_recon * region_loss)
                + af * cfg.w_contrast * contrast_loss
            )

            guard_finite("total_loss", loss)

        # backward / step
        if train:
            optimizer.zero_grad(set_to_none=True)
            if cfg.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                for n, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise SystemExit(f"[Grad NaN/Inf] {n} grad is non-finite.")
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                for n, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise SystemExit(f"[Grad NaN/Inf] {n} grad is non-finite.")
                optimizer.step()

            if ema is not None:
                ema.update(model)

        acc = accuracy_top1(logits, y)
        bs = batch_img.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_n += bs

        dt = time.perf_counter() - t_step_start
        if step_time_ema is None:
            step_time_ema = dt
        else:
            step_time_ema = 0.9 * step_time_ema + 0.1 * dt
        if eta_state is not None:
            eta_state["train_step_ema" if train else "val_step_ema"] = step_time_ema

        step += 1

        if cfg.log_every_n_steps > 0 and (step % cfg.log_every_n_steps == 0):
            steps_done = bidx + 1
            steps_remain = max(0, len(loader) - steps_done)
            eta_phase_sec = (step_time_ema or 0.0) * steps_remain
            logger.info(
                f"[{phase}] step={step:06d} | "
                f"loss={loss.item():.4f} | acc={acc:.3f} | "
                f"(cls={cls_loss:.4f}, aux_cls={aux_cls_loss:.4f}, "
                f"g_recon={graph_loss:.4f} [x={x_loss:.4f}, pos={pos_loss:.4f}, r={r_loss:.4f}], "
                f"img={img_loss:.4f}, region={region_loss:.4f}, contrast={contrast_loss:.4f}) | "
                f"aux_factor={af:.3f} | ETA(phase)={_fmt_eta(eta_phase_sec)}"
            )

    avg_loss = total_loss / max(1, total_n)
    avg_acc = total_acc / max(1, total_n)
    logger.info(f"[{phase}] avg_loss={avg_loss:.4f} | avg_acc={avg_acc:.4f} | n={total_n} | aux_factor={af:.3f}")
    return avg_loss, avg_acc, step


# =========================== Main ===========================

def main(cfg=None):
    cfg = cfg or TrainConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    # 设备 & 随机数
    device = select_device(gpu_index=0, logger=logger)
    torch.manual_seed(cfg.split_seed)
    random.seed(cfg.split_seed)

    # DataModule
    dm = WoodFusionDataModule(cfg, device, logger).setup()
    loader_train, loader_val, loader_test = dm.build_loaders()
    node_feat_dim, edge_feat_dim, out_classes = dm.infer_dims()

    model = TriModalSwinFusionModel(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        out_classes=out_classes,
        cfg=cfg,
    ).to(device)

    # ✅ dummy forward：确认 Swin 输出维度和多尺度尺寸
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, cfg.image_size[0], cfg.image_size[1], device=device)
        zI, feat_last, feats_stages = model.swin_enc(dummy)
        logger.info(f"[Swin] dummy zI={tuple(zI.shape)} feat_last={tuple(feat_last.shape)} "
                    f"stages={[tuple(f.shape) for f in feats_stages]}")
    model.train()

    if DEBUG_NAN:
        torch.autograd.set_detect_anomaly(True)

    # 类别权重
    class_w = None
    if cfg.use_class_weight:
        w, counts, n, info = compute_class_weights_fusion_robust(
            num_classes=out_classes,
            ds=dm.ds_train,
            loader=loader_train,
            class_to_idx=getattr(dm, "class_to_idx", None),
            logger=logger,
            clamp_max=10.0,
        )
        logger.info(f"[class] src={info['source']} n={n} counts={counts.tolist()} | weight={w.tolist()}")

        if n > 0 and counts.sum().item() > 0 and torch.isfinite(w).all() and w.max().item() > 0:
            class_w = w.to(device)
        else:
            logger.info("[class] disable class_weight (invalid)")

    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=cfg.label_smoothing)

    # 分层学习率（backbone 小 lr）
    backbone_params = list(model.swin_enc.base_model.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    pg_backbone = [p for p in model.parameters() if id(p) in backbone_ids]
    pg_head = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": pg_backbone, "lr": cfg.lr_backbone},
            {"params": pg_head,     "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-3,
        threshold_mode="rel",
        cooldown=1,
        min_lr=1e-6,
        verbose=True,
    )

    device_type = "cuda" if (isinstance(device, torch.device) and device.type == "cuda") else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=cfg.amp)

    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    logger.info(f"[split] ds_train={len(dm.ds_train)} | ds_val={len(dm.ds_val)}")
    logger.info(
        f"=== Train start: epochs={cfg.epochs}, "
        f"train={len(dm.ds_train)}, val={len(dm.ds_val)}, device={device}, "
        f"drop_path={cfg.swin_drop_path}, ema={cfg.use_ema}, "
        f"region_base_stage={cfg.region_base_stage}, region_recon_stage={cfg.region_recon_stage}, "
        f"region_multiscale={cfg.region_use_multiscale} ==="
    )

    best_val_acc = -1.0
    global_step = 0
    history = []
    hist_csv = os.path.join(cfg.out_dir, "training_history.csv")
    eta_state = {}

    for epoch in range(1, cfg.epochs + 1):
        logger.info(f"\n===== Epoch {epoch}/{cfg.epochs} =====")

        # 冻结/解冻 backbone（注意：现在 backbone 在 base_model）
        if epoch <= cfg.freeze_swin_epochs:
            set_requires_grad(model.swin_enc.base_model, False)
            logger.info(f"[freeze] Swin backbone frozen (epoch {epoch}/{cfg.freeze_swin_epochs})")
        else:
            set_requires_grad(model.swin_enc.base_model, True)

        train_loss, train_acc, global_step = run_one_epoch(
            model=model,
            loader=loader_train,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            criterion=criterion,
            cfg=cfg,
            step_base=global_step,
            train=True,
            epoch=epoch,
            ema=ema,
            eta_state=eta_state,
        )

        with torch.no_grad():
            if ema is not None:
                with ema.apply(model):
                    val_loss, val_acc, _ = run_one_epoch(
                        model=model,
                        loader=loader_val,
                        optimizer=optimizer,
                        scaler=scaler,
                        device=device,
                        criterion=criterion,
                        cfg=cfg,
                        step_base=0,
                        train=False,
                        epoch=epoch,
                        ema=None,
                        eta_state=eta_state,
                    )
            else:
                val_loss, val_acc, _ = run_one_epoch(
                    model=model,
                    loader=loader_val,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    criterion=criterion,
                    cfg=cfg,
                    step_base=0,
                    train=False,
                    epoch=epoch,
                    ema=None,
                    eta_state=eta_state,
                )

        curr_lr0 = optimizer.param_groups[0]["lr"]
        append_history(history, epoch, "train", train_loss, train_acc, curr_lr0)
        append_history(history, epoch, "val", val_loss, val_acc, curr_lr0)
        save_history_csv(hist_csv, history)
        plot_history_curves(history, cfg.out_dir, prefix="history")

        scheduler.step(val_acc)

        curr_lr0 = optimizer.param_groups[0]["lr"]
        curr_lr1 = optimizer.param_groups[1]["lr"]

        # 保存 last
        ckpt_last = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "cfg": cfg.__dict__,
            "best_val_acc": best_val_acc,
            "ema": (ema.shadow if ema is not None else None),
        }
        torch.save(ckpt_last, os.path.join(cfg.out_dir, cfg.save_last_name))

        # 保存 best（优先保存 EMA）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if ema is not None:
                torch.save(ema.shadow, os.path.join(cfg.out_dir, cfg.save_best_name))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.out_dir, cfg.save_best_name))
            logger.info(f"[ckpt] ✅ new best val_acc={best_val_acc:.4f} @ epoch={epoch} (saved)")

        logger.info(
            f"[epoch] {epoch:03d} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"lr(backbone)={curr_lr0:.6f} lr(head)={curr_lr1:.6f}"
        )

    logger.info("✅ Training finished.")

    # ===== 加载 best（EMA 或普通权重）=====
    best_path = os.path.join(cfg.out_dir, cfg.save_best_name)
    if os.path.isfile(best_path):
        best_sd = torch.load(best_path, map_location=device)
        model.load_state_dict(best_sd, strict=False)
        logger.info(f"[load] best loaded from {best_path}")

    logger.info("Use run.py evaluate for held-out visual evaluation and run.py fuse for score fusion.")


if __name__ == "__main__":
    main()
