# -*- coding: utf-8 -*-
"""
stacking_fusion_test.py

"""

import os
import re
import json
import csv
import math
import argparse
import importlib
from types import SimpleNamespace

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# headless 服务器建议
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== 你项目里已有的工具 ======
from FusionDataset import logger
from util.fusion_utils import select_device
from util.data_module import WoodFusionDataModule


# ------------------------- IO & small utils -------------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def _safe_get(d, k, default=None):
    try:
        return d.get(k, default)
    except Exception:
        return default

def _normalize_path(p: str) -> str:
    if p is None:
        return ""
    return str(p).replace("\\", "/").strip()

def _maybe_get_img_path(meta_item):
    if meta_item is None:
        return ""
    if isinstance(meta_item, str):
        return meta_item
    if isinstance(meta_item, dict):
        return str(_safe_get(meta_item, "img_path", _safe_get(meta_item, "path", "")) or "")
    return ""

def _strip_prefix_in_state_dict(sd: dict):
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for p in ("module.", "model.", "backbone."):
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd

def _load_cfg_from_run(run_dir: str):
    cfg_path = os.path.join(run_dir, "cfg.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"[cfg] not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_cfg_object(cfg_dict: dict):
    return SimpleNamespace(**cfg_dict)

def _infer_class_names(dm, out_classes: int):
    idx_to_class = getattr(dm, "idx_to_class", None)
    if isinstance(idx_to_class, dict) and len(idx_to_class) >= out_classes:
        return [str(idx_to_class[i]) for i in range(out_classes)]
    class_to_idx = getattr(dm, "class_to_idx", None)
    if isinstance(class_to_idx, dict) and len(class_to_idx) >= out_classes:
        inv = {int(v): str(k) for k, v in class_to_idx.items()}
        return [inv.get(i, str(i)) for i in range(out_classes)]
    return [str(i) for i in range(out_classes)]

def _make_perm(out_classes: int, swap_5_6: bool):
    perm = list(range(out_classes))
    if swap_5_6 and out_classes >= 6:
        perm[4], perm[5] = perm[5], perm[4]  # 位置5↔6（1-based）
    return perm

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, C: int):
    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < C and 0 <= p < C:
            cm[t, p] += 1
    return cm

def _metrics_from_cm(cm: np.ndarray, eps: float = 1e-12):
    C = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)

    precision = tp / (pred_count + eps)
    recall = tp / (support + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    macro = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
    }
    total = float(np.sum(support) + eps)
    weighted = {
        "precision": float(np.sum(precision * support) / total),
        "recall": float(np.sum(recall * support) / total),
        "f1": float(np.sum(f1 * support) / total),
    }
    return {
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
        "macro": macro,
        "weighted": weighted,
    }

def _write_csv(path: str, rows: list, header: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


# ------------------------- RF csv fallback (optional) -------------------------

def load_rf_csv_map(rf_csv: str, num_classes: int, logger=None):
    """
    读取 rf_csv，把每个样本映射到一个 [C] 向量（prob 或 logits）。
    会尝试自动识别 key 列（img_path/path/file/filename）。
    会尝试识别概率列（p_/prob），否则取最后 C 列。
    """
    if not rf_csv:
        return None
    if not os.path.isfile(rf_csv):
        raise FileNotFoundError(f"[rf_csv] not found: {rf_csv}")

    with open(rf_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if len(headers) < (num_classes + 1):
            raise RuntimeError(f"[rf_csv] columns too few: {len(headers)}")

        key_col = None
        for kc in ("img_path", "path", "file", "filename", "image", "img"):
            if kc in headers:
                key_col = kc
                break
        if key_col is None:
            key_col = headers[0]

        prob_cols = [h for h in headers if h.lower().startswith("p_") or "prob" in h.lower()]
        if len(prob_cols) < num_classes:
            prob_cols = headers[-num_classes:]

        rf_map = {}
        n = 0
        for row in reader:
            k = _normalize_path(row.get(key_col, ""))
            if not k:
                continue
            vec = []
            for c in prob_cols[:num_classes]:
                try:
                    vec.append(float(row[c]))
                except Exception:
                    vec.append(0.0)
            rf_map[k] = np.asarray(vec, dtype=np.float32)
            n += 1

    if logger:
        logger.info(f"[rf_csv] loaded {n} rows | key_col={key_col} | prob_cols={prob_cols[:num_classes]}")
    return rf_map

def _is_prob_matrix(x: torch.Tensor, eps: float = 1e-4) -> bool:
    if x is None or (not torch.is_tensor(x)) or x.numel() == 0:
        return False
    if x.min().item() < -eps:
        return False
    if x.max().item() > 1.0 + 1e-3:
        return False
    s = x.sum(dim=1)
    return bool(torch.allclose(s, torch.ones_like(s), atol=5e-2, rtol=0.0))

def _rf_to_logits(batch_rf: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    如果 batch_rf 看起来像概率（0..1 且每行和≈1），转为 log-prob 当 logits；
    否则认为它本身就是 logits。
    """
    if _is_prob_matrix(batch_rf):
        return torch.log(batch_rf.clamp_min(eps))
    return batch_rf


# ------------------------- Confusion Matrix Plot (count + row%) -------------------------

def plot_cm_count_and_rowpct(cm: np.ndarray, class_names, save_path: str):
    cm = np.asarray(cm, dtype=np.int64)
    C = cm.shape[0]
    row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
    cm_norm = cm.astype(np.float64) / np.clip(row_sum, 1.0, None)

    fig_w = max(6, 0.75 * C)
    fig_h = max(5, 0.70 * C)
    plt.figure(figsize=(fig_w, fig_h))

    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Greens", vmin=0.0, vmax=1.0)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(np.arange(C), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(C), class_names)

    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, C, 1), minor=True)
    ax.set_yticks(np.arange(-.5, C, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    thr = 0.50
    for i in range(C):
        for j in range(C):
            count = int(cm[i, j])
            pct = cm_norm[i, j] * 100.0
            color = "white" if cm_norm[i, j] >= thr else "black"
            plt.text(j, i, f"{count}\n{pct:.2f}%", ha="center", va="center", fontsize=6, color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ------------------------- Stacking Heads -------------------------

class DiagStackingHead(nn.Module):
    """
    每个类、每个模态一个系数：
      fused_c = alpha_c * gi_c + beta_c * rf_c + bias_c
    """
    def __init__(self, num_classes: int, init_alpha=1.0, init_beta=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_classes,), float(init_alpha)))
        self.beta  = nn.Parameter(torch.full((num_classes,), float(init_beta)))
        self.bias  = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits_gi: torch.Tensor, logits_rf: torch.Tensor) -> torch.Tensor:
        return logits_gi * self.alpha + logits_rf * self.beta + self.bias


class FullStackingHead(nn.Module):
    """
    更强但更易过拟合：输入 concat([logits_gi, logits_rf]) -> Linear(2C->C)
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(2 * num_classes, num_classes)

    def forward(self, logits_gi: torch.Tensor, logits_rf: torch.Tensor) -> torch.Tensor:
        x = torch.cat([logits_gi, logits_rf], dim=1)
        return self.fc(x)


def train_stacking_head(
    logits_gi_val: np.ndarray,
    logits_rf_val: np.ndarray,
    y_val: np.ndarray,
    mode: str,
    num_classes: int,
    device: torch.device,
    epochs: int = 300,
    lr: float = 5e-2,
    weight_decay: float = 1e-3,
    batch_size: int = 1024,
    seed: int = 2025,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if mode == "diag":
        head = DiagStackingHead(num_classes=num_classes).to(device)
    elif mode == "full":
        head = FullStackingHead(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown stacking_mode: {mode}")

    Xg = torch.from_numpy(logits_gi_val).float()
    Xr = torch.from_numpy(logits_rf_val).float()
    y  = torch.from_numpy(y_val).long()

    ds = torch.utils.data.TensorDataset(Xg, Xr, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = -1.0
    best_sd = None

    for ep in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_n = 0

        for bg, br, by in dl:
            bg = bg.to(device, non_blocking=True)
            br = br.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            logits = head(bg, br)
            loss = F.cross_entropy(logits, by)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = int(by.numel())
            total_loss += float(loss.item()) * bs
            total_n += bs

        # eval on full val
        head.eval()
        with torch.no_grad():
            bg = Xg.to(device)
            br = Xr.to(device)
            by = y.to(device)
            fused = head(bg, br)
            pred = torch.argmax(fused, dim=1)
            acc = float((pred == by).float().mean().item())

        if acc > best_acc:
            best_acc = acc
            best_sd = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        if ep == 1 or ep % 20 == 0 or ep == epochs:
            avg_loss = total_loss / max(1, total_n)
            logger.info(f"[stacking][{mode}] epoch={ep:03d}/{epochs} loss={avg_loss:.6f} val_acc={acc:.6f} best={best_acc:.6f}")

    if best_sd is not None:
        head.load_state_dict(best_sd, strict=True)
    logger.info(f"[stacking][{mode}] done. best_val_acc={best_acc:.6f}")
    return head, best_acc


# ------------------------- Extract logits from loaders -------------------------

@torch.no_grad()
def extract_logits_from_loader(
    gi_model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    rf_map: dict = None,
    amp: bool = False,
):
    gi_model.eval()
    device_type = "cuda" if (isinstance(device, torch.device) and device.type == "cuda") else "cpu"

    all_y = []
    all_img_path = []
    all_gi = []
    all_rf = []

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 4:
            continue
        batch_graph, batch_img, batch_rf, meta = batch[0], batch[1], batch[2], batch[3]
        if batch_graph is None:
            continue

        batch_img = batch_img.to(device, non_blocking=True)
        try:
            batch_graph = batch_graph.to(device)
        except Exception:
            pass

        y = batch_graph.y.view(-1).to(device)

        # rf logits：优先用 batch_rf；如果没有就用 rf_csv + meta.img_path 查
        if batch_rf is None and rf_map is not None:
            # 用 meta 查
            paths = []
            if isinstance(meta, (list, tuple)):
                for mi in meta:
                    paths.append(_normalize_path(_maybe_get_img_path(mi)))
            rf_np = []
            for p in paths:
                v = rf_map.get(p, None)
                if v is None:
                    # 再试 basename 匹配
                    bn = os.path.basename(p)
                    v = rf_map.get(bn, None)
                if v is None:
                    v = np.zeros((num_classes,), dtype=np.float32)
                rf_np.append(v)
            batch_rf = torch.from_numpy(np.stack(rf_np, axis=0)).float()

        if batch_rf is None:
            raise RuntimeError("[rf] batch_rf is None and rf_map is None -> no RF logits available.")

        if torch.is_tensor(batch_rf):
            batch_rf = batch_rf.to(device, non_blocking=True).float()
        else:
            batch_rf = torch.as_tensor(batch_rf, device=device, dtype=torch.float32)

        rf_logits = _rf_to_logits(batch_rf)  # prob -> logprob ；logits -> logits

        with torch.amp.autocast(device_type=device_type, enabled=bool(amp)):
            gi_logits = gi_model(batch_graph, batch_img, return_details=False)

        if gi_logits.shape[1] != num_classes:
            raise RuntimeError(f"[gi] logits dim mismatch: {tuple(gi_logits.shape)} vs C={num_classes}")
        if rf_logits.shape[1] != num_classes:
            raise RuntimeError(f"[rf] logits dim mismatch: {tuple(rf_logits.shape)} vs C={num_classes}")

        bs = int(y.numel())

        all_y.append(y.detach().cpu().numpy().astype(np.int64))
        all_gi.append(gi_logits.detach().cpu().numpy().astype(np.float32))
        all_rf.append(rf_logits.detach().cpu().numpy().astype(np.float32))

        if isinstance(meta, (list, tuple)) and len(meta) == bs:
            for mi in meta:
                all_img_path.append(_maybe_get_img_path(mi))
        else:
            all_img_path.extend([""] * bs)

    y_np = np.concatenate(all_y, axis=0)
    gi_np = np.concatenate(all_gi, axis=0)
    rf_np = np.concatenate(all_rf, axis=0)

    return y_np, gi_np, rf_np, all_img_path


# ------------------------- Evaluate fused on loader -------------------------

@torch.no_grad()
def evaluate_fused(
    gi_model: nn.Module,
    head: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    rf_map: dict = None,
    amp: bool = False,
):
    gi_model.eval()
    head.eval()
    device_type = "cuda" if (isinstance(device, torch.device) and device.type == "cuda") else "cpu"

    criterion = nn.CrossEntropyLoss()

    all_true, all_pred, all_prob = [], [], []
    all_img_path = []
    total_loss, total_n, correct = 0.0, 0, 0

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 4:
            continue
        batch_graph, batch_img, batch_rf, meta = batch[0], batch[1], batch[2], batch[3]
        if batch_graph is None:
            continue

        batch_img = batch_img.to(device, non_blocking=True)
        try:
            batch_graph = batch_graph.to(device)
        except Exception:
            pass
        y = batch_graph.y.view(-1).to(device)

        # rf
        if batch_rf is None and rf_map is not None:
            paths = []
            if isinstance(meta, (list, tuple)):
                for mi in meta:
                    paths.append(_normalize_path(_maybe_get_img_path(mi)))
            rf_np = []
            for p in paths:
                v = rf_map.get(p, None)
                if v is None:
                    bn = os.path.basename(p)
                    v = rf_map.get(bn, None)
                if v is None:
                    v = np.zeros((num_classes,), dtype=np.float32)
                rf_np.append(v)
            batch_rf = torch.from_numpy(np.stack(rf_np, axis=0)).float()

        if batch_rf is None:
            raise RuntimeError("[rf] batch_rf is None and rf_map is None -> no RF logits available.")

        if torch.is_tensor(batch_rf):
            batch_rf = batch_rf.to(device, non_blocking=True).float()
        else:
            batch_rf = torch.as_tensor(batch_rf, device=device, dtype=torch.float32)
        rf_logits = _rf_to_logits(batch_rf)

        with torch.amp.autocast(device_type=device_type, enabled=bool(amp)):
            gi_logits = gi_model(batch_graph, batch_img, return_details=False)
            fused_logits = head(gi_logits, rf_logits)
            loss = criterion(fused_logits, y)

        probs = F.softmax(fused_logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        bs = int(y.numel())
        total_loss += float(loss.item()) * bs
        total_n += bs
        correct += int((pred == y).sum().item())

        all_true.append(y.detach().cpu().numpy().astype(np.int64))
        all_pred.append(pred.detach().cpu().numpy().astype(np.int64))
        all_prob.append(probs.detach().cpu().numpy().astype(np.float32))

        if isinstance(meta, (list, tuple)) and len(meta) == bs:
            for mi in meta:
                all_img_path.append(_maybe_get_img_path(mi))
        else:
            all_img_path.extend([""] * bs)

    if total_n <= 0:
        raise RuntimeError("[eval] got 0 samples from loader.")

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    prob   = np.concatenate(all_prob, axis=0)

    avg_loss = total_loss / max(1, total_n)
    acc = float(correct) / float(max(1, total_n))
    cm = _confusion_matrix(y_true, y_pred, C=num_classes)
    met = _metrics_from_cm(cm)

    return {
        "n": int(total_n),
        "loss": float(avg_loss),
        "acc": float(acc),
        "y_true": y_true,
        "y_pred": y_pred,
        "prob": prob,
        "img_path": all_img_path,
        "cm": cm,
        "metrics": met,
    }


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--run_dir", type=str, help="GI run_dir with cfg.json", default="outputs/ckeswin")
    ap.add_argument("--weights", type=str, default="best.pt", help="GI best.pt (relative or abs)")
    ap.add_argument("--gi_model_module", type=str,default="train_tri_modal_swin_fusion_v3_2",help="python module name that defines TriModalSwinFusionModel, e.g. train_tri_modal_swin_fusion_v3_acc95_region_hires")

    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--val_file", type=str, default="", help="override cfg.val_file / cfg.test_file (depends on your DM)")
    ap.add_argument("--test_file", type=str, default="", help="override cfg.test_file")
    ap.add_argument("--batch_size", type=int, default=0)

    ap.add_argument("--rf_csv", type=str, default="", help="fallback rf csv if dm doesn't provide batch_rf")
    ap.add_argument("--out_subdir", type=str, default="stacking_test")

    ap.add_argument("--stacking_mode", type=str, default="diag", choices=["diag"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--swap_5_6", type=int, default=1)

    args = ap.parse_args()

    run_dir = args.run_dir
    w_path = args.weights
    if not os.path.isabs(w_path):
        w_path = os.path.join(run_dir, w_path)
    if not os.path.isfile(w_path):
        raise FileNotFoundError(f"[weights] not found: {w_path}")

    # ---- cfg ----
    cfg_dict = _load_cfg_from_run(run_dir)

    # 尽量兼容：如果你 DM 只有 test_file，就用 test_file 当 val/test 的入口
    if args.batch_size and args.batch_size > 0:
        cfg_dict["batch_size"] = int(args.batch_size)

    # 这里我们先用 dm.build_loaders() 的 loader_val/loader_test；
    # 如果你强制指定 val_file/test_file，而 DM 不支持两个入口，则可在下面按需调整 cfg key。
    if args.test_file:
        cfg_dict["test_file"] = args.test_file
    if args.val_file:
        # 你 DM 若有 val_file 就写进去；否则很多实现会用 test_file 做 eval split，这里也写一份 val_file 供你 DM 取用
        cfg_dict["val_file"] = args.val_file

    if args.rf_csv:
        cfg_dict["rf_csv"] = args.rf_csv
    cfg = _build_cfg_object(cfg_dict)

    # ---- device ----
    device = select_device(gpu_index=int(args.gpu), logger=logger)

    # ---- DataModule ----
    # ⚠️ 重要：setup() 很多实现不 return self，所以分两行写最稳
    dm = WoodFusionDataModule(cfg, device, logger)
    dm.setup()
    loader_train, loader_val, loader_test = dm.build_loaders()
    if loader_train is None or len(loader_train.dataset) == 0:
        raise ValueError("The paper fusion protocol requires both training and validation pairs.")

    if loader_val is None or len(loader_val.dataset) == 0:
        raise RuntimeError("[dm] loader_val is None. Check your DM split construction.")
    if loader_test is None or len(loader_test.dataset) == 0:
        raise ValueError("A nonempty held-out test split is required.")

    node_feat_dim, edge_feat_dim, out_classes = dm.infer_dims()
    class_names = _infer_class_names(dm, out_classes)

    # ---- RF csv fallback ----
    rf_map = None
    rf_csv = args.rf_csv or getattr(cfg, "rf_csv", "")
    if rf_csv:
        try:
            rf_map = load_rf_csv_map(rf_csv, out_classes, logger=logger)
        except Exception as e:
            logger.info(f"[rf_csv] load failed, will rely on batch_rf if available. err={e}")

    # ---- import GI model ----
    gi_mod = importlib.import_module(args.gi_model_module)
    if not hasattr(gi_mod, "TriModalSwinFusionModel"):
        raise RuntimeError(f"[gi_model_module] {args.gi_model_module} has no TriModalSwinFusionModel")
    TriModalSwinFusionModel = gi_mod.TriModalSwinFusionModel

    gi_model = TriModalSwinFusionModel(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        out_classes=out_classes,
        cfg=cfg,
    ).to(device)

    # ---- load weights ----
    obj = torch.load(w_path, map_location="cpu")
    if isinstance(obj, dict):
        # 兼容：ckpt dict / ema dict / state_dict dict
        if "ema" in obj and isinstance(obj["ema"], dict):
            sd = obj["ema"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        elif "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        else:
            sd = obj
    else:
        sd = obj

    sd = _strip_prefix_in_state_dict(sd)
    missing, unexpected = gi_model.load_state_dict(sd, strict=False)
    logger.info(f"[load GI] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        logger.info(f"[load GI] missing sample: {missing[:20]}")
    if len(unexpected) > 0:
        logger.info(f"[load GI] unexpected sample: {unexpected[:20]}")

    # ---- output dir ----
    out_dir = _ensure_dir(os.path.join(run_dir, args.out_subdir))
    with open(os.path.join(out_dir, "stacking_cfg_used.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run_dir": run_dir,
            "weights": w_path,
            "gi_model_module": args.gi_model_module,
            "stacking_mode": args.stacking_mode,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "swap_5_6": int(args.swap_5_6),
            "rf_csv": rf_csv,
            "cfg": cfg_dict,
        }, f, ensure_ascii=False, indent=2)

    # ---- 取 amp 开关 ----
    amp = bool(getattr(cfg, "amp", False))

    # =========================
    # 1) 在 TRAIN + VAL 抽取 logits（用于拟合 stacking head）
    # =========================
    if loader_train is None:
        logger.info("[dm] loader_train is None -> fallback: fit on VAL only")
        y_tr = gi_tr = rf_tr = None
    else:
        logger.info("[stage] extract logits on TRAIN ...")
        y_tr, gi_tr, rf_tr, _ = extract_logits_from_loader(
            gi_model=gi_model,
            loader=loader_train,
            device=device,
            num_classes=out_classes,
            rf_map=rf_map,
            amp=amp,
        )
        logger.info(f"[train] N={len(y_tr)} gi={gi_tr.shape} rf={rf_tr.shape}")

    logger.info("[stage] extract logits on VAL ...")
    y_val, gi_val, rf_val, _ = extract_logits_from_loader(
        gi_model=gi_model,
        loader=loader_val,
        device=device,
        num_classes=out_classes,
        rf_map=rf_map,
        amp=amp,
    )
    logger.info(f"[val] N={len(y_val)} gi={gi_val.shape} rf={rf_val.shape}")

    # 拼接 train+val 作为 stacking 拟合集
    if y_tr is not None:
        y_fit  = np.concatenate([y_tr,  y_val], axis=0)
        gi_fit = np.concatenate([gi_tr, gi_val], axis=0)
        rf_fit = np.concatenate([rf_tr, rf_val], axis=0)
        logger.info(f"[fit=train+val] N={len(y_fit)} (train={len(y_tr)} val={len(y_val)})")
    else:
        y_fit, gi_fit, rf_fit = y_val, gi_val, rf_val
        logger.info(f"[fit=val_only] N={len(y_fit)}")

    # =========================
    # 2) 训练 Stacking head（在 train+val 上拟合）
    # =========================
    logger.info("[stage] train stacking head on FIT (train+val) ...")
    head, best_fit_acc = train_stacking_head(
        logits_gi_val=gi_fit,
        logits_rf_val=rf_fit,
        y_val=y_fit,
        mode=args.stacking_mode,
        num_classes=out_classes,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=2048,
        seed=int(getattr(cfg, "split_seed", 2025)),
    )

    # 为了保持你后面 summary 字段名不改，这里沿用 best_val_acc 变量名
    best_val_acc = float(best_fit_acc)

    # 保存 stacking head
    torch.save(head.state_dict(), os.path.join(out_dir, "stacking_head.pt"))

    # 若是 diag，导出每类权重（你要的“每个类、每个模态信多少”）
    if args.stacking_mode == "diag":
        alpha = head.alpha.detach().cpu().numpy()
        beta  = head.beta.detach().cpu().numpy()
        bias  = head.bias.detach().cpu().numpy()
        rows = []
        for i in range(out_classes):
            rows.append([i, class_names[i], float(alpha[i]), float(beta[i]), float(bias[i])])
        _write_csv(os.path.join(out_dir, "stacking_weights.csv"), rows, ["class_idx", "class_name", "alpha_gi", "beta_rf", "bias"])

    # =========================
    # 3) 在测试集推理融合结果
    # =========================
    logger.info("[stage] evaluate FUSED on TEST ...")
    res = evaluate_fused(
        gi_model=gi_model,
        head=head,
        loader=loader_test,
        device=device,
        num_classes=out_classes,
        rf_map=rf_map,
        amp=amp,
    )

    n = res["n"]
    loss = res["loss"]
    acc = res["acc"]
    cm = res["cm"]
    met = res["metrics"]
    macro = met["macro"]
    weighted = met["weighted"]

    logger.info(f"[test fused] n={n} loss={loss:.6f} acc={acc:.6f} macro_f1={macro['f1']:.6f} weighted_f1={weighted['f1']:.6f}")

    # summary
    summary = {
        "run_dir": run_dir,
        "weights_gi": w_path,
        "rf_csv": rf_csv,
        "stacking_mode": args.stacking_mode,
        "best_val_acc_stacking": float(best_val_acc),
        "n": int(n),
        "loss": float(loss),
        "acc": float(acc),
        "macro_precision": float(macro["precision"]),
        "macro_recall": float(macro["recall"]),
        "macro_f1": float(macro["f1"]),
        "weighted_precision": float(weighted["precision"]),
        "weighted_recall": float(weighted["recall"]),
        "weighted_f1": float(weighted["f1"]),
    }
    with open(os.path.join(out_dir, "test_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _write_csv(
        os.path.join(out_dir, "test_metrics_summary.csv"),
        rows=[[
            summary["n"], summary["loss"], summary["acc"],
            summary["macro_precision"], summary["macro_recall"], summary["macro_f1"],
            summary["weighted_precision"], summary["weighted_recall"], summary["weighted_f1"],
            summary["best_val_acc_stacking"],
        ]],
        header=[
            "n", "loss", "acc",
            "macro_precision", "macro_recall", "macro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1",
            "best_val_acc_stacking"
        ]
    )

    # per-class（可按显示顺序对调 5/6）
    perm = _make_perm(out_classes, bool(args.swap_5_6))
    class_names_disp = [class_names[i] for i in perm]

    pc = met["per_class"]
    per_rows = []
    for i in perm:
        per_rows.append([
            i,
            class_names[i],
            int(pc["support"][i]),
            float(pc["precision"][i]),
            float(pc["recall"][i]),
            float(pc["f1"][i]),
        ])
    _write_csv(os.path.join(out_dir, "test_per_class.csv"), per_rows,
               ["class_idx", "class_name", "support", "precision", "recall", "f1"])

    # confusion_matrix.csv（按显示顺序对调 5/6）
    cm_disp = cm[np.ix_(perm, perm)]
    with open(os.path.join(out_dir, "confusion_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names_disp)
        for r in range(out_classes):
            w.writerow([class_names_disp[r]] + cm_disp[r, :].tolist())

    # confusion_matrix.png（count + row% + 白绿配色）
    plot_cm_count_and_rowpct(cm_disp, class_names_disp, os.path.join(out_dir, "confusion_matrix.png"))

    # predictions.csv（融合后的概率）
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    prob   = res["prob"]
    img_paths = res["img_path"]

    header = [
        "index", "img_path",
        "y_true", "true_name",
        "y_pred", "pred_name",
        "correct",
        "prob_pred",
    ] + [f"p_{i}_{class_names[i]}" for i in range(out_classes)]

    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(y_true)):
            t = int(y_true[i])
            p = int(y_pred[i])
            row_prob = prob[i].tolist()
            prob_pred = float(row_prob[p]) if 0 <= p < out_classes else 0.0
            w.writerow([
                i,
                img_paths[i] if i < len(img_paths) else "",
                t, class_names[t] if 0 <= t < out_classes else str(t),
                p, class_names[p] if 0 <= p < out_classes else str(p),
                int(t == p),
                prob_pred,
                *row_prob
            ])

    # classification_report.txt（简版）
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Split: test\n")
        f.write(f"N: {n}\n")
        f.write(f"Loss: {loss:.6f}\n")
        f.write(f"Acc: {acc:.6f}\n")
        f.write(f"Macro P/R/F1: {macro['precision']:.6f} / {macro['recall']:.6f} / {macro['f1']:.6f}\n")
        f.write(f"Weighted P/R/F1: {weighted['precision']:.6f} / {weighted['recall']:.6f} / {weighted['f1']:.6f}\n")
        f.write(f"Stacking mode: {args.stacking_mode}\n")
        f.write(f"Best val acc (stacking fit): {best_val_acc:.6f}\n\n")
        f.write("Per-class (display order):\n")
        f.write("idx\tname\tsupport\tprecision\trecall\tf1\n")
        for r in per_rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\n")

    logger.info(f"[done] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
