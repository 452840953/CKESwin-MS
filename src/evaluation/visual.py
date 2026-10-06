# -*- coding: utf-8 -*-
"""
test_trimodal_swin_fusion.py

用于测试 TriModalSwinFusionModel（Graph + Swin Global + Region Mask 多尺度池化）。

✅ 基于你原来的 test_pure_deit.py 结构尽量保留格式，只做“适配训练代码”的必要改动：
- 模型改为从训练脚本动态导入 TriModalSwinFusionModel
- forward 改为 model(batch_graph, batch_img, return_details=False) -> logits
- 权重加载兼容：
  1) best.pt 直接是 state_dict（训练脚本保存 best 就是这个）
  2) last.pt 这种 dict：{"model":..., "ema":..., ...}
- cfg.json 类型兼容：tuple 在 json 里会变 list，这里做回填（如 image_size、region_stage_indices）

保留：
- predictions.csv 保存格式
- confusion_matrix 绘图样式（Greens + grid + count + row%）
- 可视化时交换 class 4/5 的顺序（仅用于画图）

使用示例：
python test_trimodal_swin_fusion.py \
  --run_dir runs/TriModal_SwinFusion_ACC95_REGION_HIRES_20251215 \
  --weights best.pt \
  --gpu 0 \
  --model_module train_tri_modal_swin_fusion_v3_2 \
  --out_subdir trimodal_test_results
"""

import os
import json
import csv
import argparse
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== 项目工具 ======
from FusionDataset import logger
from util.fusion_utils import select_device
from util.data_module import WoodFusionDataModule


# ------------------------- 辅助函数 -------------------------

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _load_cfg_from_run(run_dir):
    cfg_path = os.path.join(run_dir, "cfg.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _fix_cfg_types(cfg_dict: dict) -> dict:
    """
    cfg.json 里 tuple 会变成 list，这里尽量恢复成训练时习惯的类型，减少 DataModule / 下游逻辑差异。
    """
    d = dict(cfg_dict)

    # image_size: [224,224] -> (224,224)
    if "image_size" in d and isinstance(d["image_size"], list) and len(d["image_size"]) == 2:
        d["image_size"] = (int(d["image_size"][0]), int(d["image_size"][1]))

    # region_stage_indices: [0,1,2,3] -> (0,1,2,3)
    if "region_stage_indices" in d and isinstance(d["region_stage_indices"], list):
        d["region_stage_indices"] = tuple(int(x) for x in d["region_stage_indices"])

    # 其它可能是 tuple 的字段（按需补）
    for k in ("region_stage_indices",):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])

    # 缺省字段兜底（以免旧 cfg 缺字段）
    d.setdefault("amp", False)
    d.setdefault("pos_format", "xy")

    return d

def _build_cfg_object(cfg_dict):
    return SimpleNamespace(**cfg_dict)

def _strip_prefix(sd: dict):
    """移除常见 DDP/Lightning 前缀，尽量温和，不做过度替换。"""
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new_sd[nk] = v
    return new_sd

def _maybe_get_path(meta_item):
    if meta_item is None:
        return ""
    if isinstance(meta_item, str):
        return meta_item
    if isinstance(meta_item, dict):
        return meta_item.get("img_path", meta_item.get("path", ""))
    return ""

def _extract_state_dict_from_checkpoint(ckpt):
    """
    兼容：
    - best.pt: 直接是 state_dict（参数名->tensor）
    - last.pt: dict，含 "ema"/"model"/"state_dict" 等
    """
    if isinstance(ckpt, dict):
        # last.pt 常见结构
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and len(ckpt["ema"]) > 0:
            return ckpt["ema"], "ema"
        if "model" in ckpt and isinstance(ckpt["model"], dict) and len(ckpt["model"]) > 0:
            return ckpt["model"], "model"
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict) and len(ckpt["state_dict"]) > 0:
            return ckpt["state_dict"], "state_dict"

        # best.pt 这种：key 很多且 value 是 Tensor
        tensor_like = 0
        for _, v in ckpt.items():
            if torch.is_tensor(v):
                tensor_like += 1
        if tensor_like > 10:
            return ckpt, "raw_state_dict"

    return ckpt, "unknown"

def plot_cm_count_and_rowpct(cm: np.ndarray, class_names, save_path: str):
    """
    绘制混淆矩阵：显示数量和行归一化百分比 (Recall方向)。
    样式参考：Greens colormap, minor grid lines.
    """
    cm = np.asarray(cm, dtype=np.int64)
    C = cm.shape[0]
    row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
    cm_norm = cm.astype(np.float64) / np.clip(row_sum, 1.0, None)

    fig_w = max(6, 0.75 * C)
    fig_h = max(5, 0.70 * C)
    plt.figure(figsize=(fig_w, fig_h))

    plt.imshow(cm_norm, interpolation="nearest", cmap="Greens", vmin=0.0, vmax=1.0)
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


# ------------------------- 核心评估循环 -------------------------

@torch.no_grad()
def evaluate_trimodal(model, loader, device, amp=False):
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    all_paths = []

    total_loss = 0.0
    total_samples = 0
    correct = 0

    criterion = nn.CrossEntropyLoss()
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    logger.info(f"Starting TriModal inference on {len(loader)} batches...")

    for batch_idx, batch in enumerate(loader):
        # DataModule 返回: (graph, img, rf, meta)
        if not isinstance(batch, (tuple, list)) or len(batch) < 4:
            continue

        batch_graph, batch_img, _, meta = batch

        if batch_graph is None:
            continue

        batch_img = batch_img.to(device, non_blocking=True)
        try:
            if hasattr(batch_graph, "to"):
                batch_graph = batch_graph.to(device)
            y = batch_graph.y.view(-1).to(device)
        except Exception:
            continue

        bs = batch_img.size(0)

        with torch.amp.autocast(device_type, enabled=amp):
            # TriModalSwinFusionModel.forward(batch_graph, batch_img, return_details=False)
            logits = model(batch_graph, batch_img, return_details=False)
            loss = criterion(logits, y)

        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * bs
        total_samples += bs
        correct += (preds == y).sum().item()

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        if isinstance(meta, (list, tuple)):
            for m in meta:
                all_paths.append(_maybe_get_path(m))
        else:
            all_paths.extend([""] * bs)

    if total_samples == 0:
        raise RuntimeError("No samples found in loader!")

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    avg_loss = total_loss / total_samples
    acc = correct / total_samples

    return {
        "loss": avg_loss,
        "acc": acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "paths": all_paths,
        "n_samples": total_samples,
    }


# ------------------------- 主程序 -------------------------

def main():
    parser = argparse.ArgumentParser(description="Test TriModal Swin Fusion Model")
    parser.add_argument("--run_dir", type=str, required=True, help="训练输出目录 (包含 cfg.json)")
    parser.add_argument("--weights", type=str, default="best.pt", help="权重文件名 (如 best.pt / last.pt)")
    parser.add_argument("--gpu", type=int, default=0)

    # 覆盖参数
    parser.add_argument("--test_file", type=str, default="", help="覆盖测试集路径")
    parser.add_argument("--model_module", type=str, default="train_tri_modal_swin_fusion_v3_2",
                        help="定义 TriModalSwinFusionModel 的文件名 (不带 .py)")
    parser.add_argument("--model_class", type=str, default="TriModalSwinFusionModel",
                        help="模型类名（默认 TriModalSwinFusionModel）")
    parser.add_argument("--out_subdir", type=str, default="trimodal_test_results", help="结果输出子目录")

    args = parser.parse_args()

    # 0) 让 importlib 更稳：把当前工作目录加入 sys.path
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # 1) 加载配置
    cfg_dict = _load_cfg_from_run(args.run_dir)
    cfg_dict = _fix_cfg_types(cfg_dict)

    # 允许命令行覆盖测试集（同时覆盖 val_file，兼容某些 DataModule 写法）
    if args.test_file:
        cfg_dict["test_file"] = args.test_file
        cfg_dict["val_file"] = args.test_file

    cfg = _build_cfg_object(cfg_dict)

    # 2) 准备设备和输出目录
    device = select_device(args.gpu, logger)
    out_dir = _ensure_dir(os.path.join(args.run_dir, args.out_subdir))
    logger.info(f"Results will be saved to: {out_dir}")

    # 3) 准备数据
    dm = WoodFusionDataModule(cfg, device, logger)
    dm.setup()

    _, _, loader_test = dm.build_loaders()
    if loader_test is None or len(loader_test.dataset) == 0:
        raise ValueError("A nonempty held-out test split is required.")

    node_dim, edge_dim, num_classes = dm.infer_dims()

    # 类别名称
    if hasattr(dm, "idx_to_class"):
        class_names = [str(dm.idx_to_class[i]) for i in range(num_classes)]
    else:
        class_names = [str(i) for i in range(num_classes)]

    # 4) 动态导入模型类
    logger.info(f"Importing model class from: {args.model_module}.{args.model_class}")
    try:
        mod_lib = importlib.import_module(args.model_module)
        ModelClass = getattr(mod_lib, args.model_class)
    except ImportError as e:
        logger.error(f"Could not import {args.model_module}. Make sure it is in PYTHONPATH / same directory.")
        raise e
    except AttributeError:
        logger.error(f"Could not find '{args.model_class}' in {args.model_module}.")
        raise

    # 5) 实例化模型（训练脚本里需要 node_dim/edge_dim/out_classes/cfg）
    model = ModelClass(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        out_classes=num_classes,
        cfg=cfg
    ).to(device)

    # 6) 加载权重
    weights_path = os.path.join(args.run_dir, args.weights)
    logger.info(f"Loading weights: {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict, sd_kind = _extract_state_dict_from_checkpoint(checkpoint)
    state_dict = _strip_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        logger.warning(f"Missing keys: {missing[:8]} ... (Total {len(missing)})")
    if len(unexpected) > 0:
        logger.warning(f"Unexpected keys: {unexpected[:8]} ... (Total {len(unexpected)})")
    logger.info(f"Loaded state_dict kind = {sd_kind}")

    # 7) 推理
    res = evaluate_trimodal(model, loader_test, device, amp=bool(getattr(cfg, "amp", False)))
    logger.info(f"Inference Done. N={res['n_samples']}, Loss={res['loss']:.4f}, Acc={res['acc']:.4f}")

    # 8) 保存结果

    # (A) Predictions CSV
    csv_path = os.path.join(out_dir, "predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["path", "true_idx", "true_name", "pred_idx", "pred_name", "correct"] + \
                 [f"prob_{c}" for c in class_names]
        writer.writerow(header)

        for i in range(res["n_samples"]):
            t = int(res["y_true"][i])
            p = int(res["y_pred"][i])
            row = [
                res["paths"][i],
                t, class_names[t] if t < len(class_names) else str(t),
                p, class_names[p] if p < len(class_names) else str(p),
                1 if t == p else 0
            ]
            row.extend(res["y_prob"][i].tolist())
            writer.writerow(row)

    # (B) Confusion Matrix（原始 + 可视化重排）
    cm = confusion_matrix(res["y_true"], res["y_pred"], labels=range(num_classes))
    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    plot_cm = cm
    plot_names = list(class_names)

    # 仅用于画图：交换索引 4/5
    if num_classes >= 6:
        logger.info("Reordering confusion matrix visualization: swap class index 4 and 5.")
        perm_idx = list(range(num_classes))
        perm_idx[4], perm_idx[5] = perm_idx[5], perm_idx[4]
        plot_cm = plot_cm[perm_idx, :]
        plot_cm = plot_cm[:, perm_idx]
        plot_names = [class_names[i] for i in perm_idx]

    plot_cm_count_and_rowpct(plot_cm, plot_names, os.path.join(out_dir, "confusion_matrix.png"))

    # (C) Classification report & metrics summary
    report_txt = classification_report(res["y_true"], res["y_pred"], target_names=class_names, digits=4)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt + "\n")

    report_dict = classification_report(
        res["y_true"], res["y_pred"], target_names=class_names, output_dict=True
    )

    summary = {
        "model_class": args.model_class,
        "model_module": args.model_module,
        "run_dir": args.run_dir,
        "weights": args.weights,
        "accuracy": float(res["acc"]),
        "loss": float(res["loss"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "per_class": {k: v for k, v in report_dict.items() if k in class_names},
    }

    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(" Test Results (TriModal Swin Fusion)")
    print("=" * 50)
    print(f" Run dir      : {args.run_dir}")
    print(f" Weights      : {args.weights}")
    print(f" Model        : {args.model_module}.{args.model_class}")
    print(f" Accuracy     : {res['acc']:.2%}")
    print(f" Macro F1     : {summary['macro_f1']:.4f}")
    print(f" Weighted F1  : {summary['weighted_f1']:.4f}")
    print("-" * 50)
    print(report_txt)
    print("=" * 50)
    print(f"Full results saved to: {out_dir}")

if __name__ == "__main__":
    main()
