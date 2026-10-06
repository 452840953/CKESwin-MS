"""Direct helpers extracted from the original shared utilities."""
import os
import re
import ntpath
from collections import Counter
from typing import List, Iterable, Optional
import numpy as np
import torch
from FusionDataset import RFCsvLookupContains
from util.gine_util import count_by_class, pretty_table


def select_device(gpu_index: int = 1, logger=None) -> str:
    """
    选择第 gpu_index 张 GPU（从 0 开始）；若不可用则回退 CPU。
    会设置默认设备：torch.cuda.set_device(gpu_index)
    """
    if torch.cuda.is_available():
        if torch.cuda.device_count() <= gpu_index:
            msg = f"请求 cuda:{gpu_index}，但仅检测到 {torch.cuda.device_count()} 张 GPU。"
            raise RuntimeError(msg)
        device = f"cuda:{gpu_index}"
        torch.cuda.set_device(gpu_index)
        name = torch.cuda.get_device_name(gpu_index)
        if logger is not None:
            logger.info(f"Using {device} - {name}")
        else:
            print(f"[Device] {device} - {name}")
        return device
    if logger is not None:
        logger.warning("CUDA 不可用，回退到 CPU。")
    else:
        print("[Device] CUDA 不可用，使用 CPU。")
    return "cpu"

def _norm_sid(s):
    if s is None:
        return None
    s = str(s).strip().replace('_', '-')
    s = re.sub(r'\s+', '', s)
    return s.upper()

def _get_img_path(di):
    # PyG Data: di.img_path
    p = getattr(di, "img_path", None)
    if p:
        return p

    # tuple/list: (graph, img, rf, meta)
    if isinstance(di, (tuple, list)):
        if len(di) >= 4 and isinstance(di[3], dict):
            p = di[3].get("img_path") or di[3].get("path")
            if p:
                return p
        if len(di) >= 1:
            p = getattr(di[0], "img_path", None)
            if p:
                return p
    return None

def _collect_specimens(ds, indices: Iterable[int]) -> List[str]:
    sids = []
    for i in indices:
        di = ds[i]
        img_path = _get_img_path(di)

        if img_path:
            fname = ntpath.basename(str(img_path))      # 关键：兼容反斜杠
            stem  = os.path.splitext(fname)[0]
            parts = re.split(r"[_-]+", stem)

            sid = parts[2] if len(parts) >= 3 else None

            # 你原来的特例
            if sid in ("028", "LSJ", "YZ") and len(parts) >= 4 and parts[3]:
                sid = f"{sid}-{parts[3]}"
        else:
            sid = None

        sid = _norm_sid(sid)
        if sid:
            sids.append(sid)

    return sids

def _log_split_report(ds, train_idx: List[int], val_idx: List[int], logger) -> None:
    # 1) 基本规模
    logger.info("=== 固定划分（基于 test.txt 标本号）报告 ===")
    logger.info(f"Train graphs: {len(train_idx)} | Val graphs: {len(val_idx)}")

    # 2) specimen 唯一数 & 交集（检测泄漏）
    try:
        train_sids = set(_collect_specimens(ds, train_idx))
        val_sids   = set(_collect_specimens(ds, val_idx))
    except Exception as e:
        logger.warning(f"[report] specimen 统计失败：{e}")
        train_sids, val_sids = set(), set()

    inter = train_sids & val_sids
    logger.info(f"Unique specimens — train: {len(train_sids)}, val: {len(val_sids)}, overlap: {len(inter)}")
    if len(inter) > 0:
        logger.warning(f"[Leakage?] 发现 {len(inter)} 个重叠 specimen: {sorted(list(inter))[:10]} ...")

    # 3) 类别分布（本地安全统计：把 y -> int）
    def _count_by_class(idxs):
        c = Counter()
        for i in idxs:
            g = ds[i]
            y = getattr(g, "y", None)
            if y is None:
                continue
            try:
                # y 可能是 Tensor([k]) / Tensor(k) / numpy / python int
                if isinstance(y, torch.Tensor):
                    y = int(y.item())
                else:
                    y = int(y)
            except Exception as ex:
                logger.warning(f"[report] 跳过无法解析的标签（type={type(y)}）：{ex}")
                continue
            c[y] += 1
        return c

    try:
        train_cnt = _count_by_class(train_idx)
        val_cnt   = _count_by_class(val_idx)
        all_keys = sorted(set(train_cnt.keys()) | set(val_cnt.keys()))

        # 若你项目里有 pretty_table，就用；否则走下面的纯文本 fallback
        try:
            table = pretty_table(
                ["Class", "Train", "Val", "Total"],
                [[k, train_cnt.get(k,0), val_cnt.get(k,0), train_cnt.get(k,0)+val_cnt.get(k,0)]
                 for k in all_keys]
            )
            logger.info("\n" + table)
        except Exception:
            lines = ["Class | Train | Val | Total", "----- | ----- | --- | -----"]
            for k in all_keys:
                tr = train_cnt.get(k, 0); va = val_cnt.get(k, 0)
                lines.append(f"{k:>5} | {tr:>5} | {va:>3} | {tr+va:>5}")
            logger.info("\n" + "\n".join(lines))

    except Exception as e:
        logger.warning(f"[report] 打印类别分布失败（可忽略）：{e}")

def check_rf_coverage(pyg_dataset, rf_csv: Optional[str], num_classes: int,
                      key_col: str = "specimen_id", max_show: int = 20, logger=None) -> None:
    if rf_csv is None:
        (logger.info if logger else print)("[coverage] rf_csv=None，跳过覆盖率体检。")
        return
    lookup = RFCsvLookupContains(rf_csv, num_classes=num_classes, key_col=key_col, seed=2025, verbose=False)
    total = len(pyg_dataset)
    hit = 0
    miss_paths = []
    for i in range(total):
        d = pyg_dataset[i]
        vec = lookup.get_by_path(d.img_path, mode="logit_mean")
        if vec is None:
            miss_paths.append(d.img_path)
        else:
            hit += 1
    rate = hit / max(1, total)
    (logger.info if logger else print)(f"[coverage] RF 命中 {hit}/{total} = {rate:.2%}")
    if miss_paths:
        (logger.info if logger else print)(f"[coverage] 未命中样例（最多列出 {max_show} 条）：")
        for p in miss_paths[:max_show]:
            (logger.info if logger else print)(f"  - {p}")

def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def _fmt_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
