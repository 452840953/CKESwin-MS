# util/split_util.py
import os
import csv
import numpy as np
import torch
from util.gine_util import load_specimens_from_txt, path_contains_any_specimen, count_by_class, pretty_table
from FusionDataset import FusionDataset

import os
import csv

def split_dataset_by_specimens(dataset, test_file, save_root, logger):
    """
    基于名单文件按“标本号(specimen)”做分组划分：
    - test_file: 验证集(val)名单（既有协议保持不变）
    - 同目录下的 'test.txt': 测试集(test)名单（若缺失则忽略测试集）

    返回:
        train_idx, val_idx, idx_to_class
    注意：
        train_idx 已排除 val 与 test（若存在）。
        测试集索引会落盘到 split_dump/test_indices.txt，可按需再读。
    """
    # === 1) 读取 val/test 标本号清单 ===
    # val（沿用原 test_file）
    val_specimens, val_specimen_to_cls = load_specimens_from_txt(test_file)
    logger.info(f"[Split] val 名单({os.path.basename(test_file)}) 标本数={len(val_specimens)}，样例(前5)：{list(val_specimens)[:5]}")

    # test（同目录下强制命名为 test.txt）
    test_dir = os.path.dirname(test_file)
    test_list_path = os.path.join(test_dir, "test.txt")
    if os.path.isfile(test_list_path):
        test_specimens, test_specimen_to_cls = load_specimens_from_txt(test_list_path)
        logger.info(f"[Split] test 名单(test.txt) 标本数={len(test_specimens)}，样例(前5)：{list(test_specimens)[:5]}")
    else:
        test_specimens, test_specimen_to_cls = set(), {}
        logger.warning(f"[Split] 未发现测试集名单：{test_list_path}，将仅划分 train/val。")

    # 名单交叉检查（避免同一标本既在 val 又在 test）
    overlap_spec = val_specimens.intersection(test_specimens)
    if overlap_spec:
        logger.error(f"[Split] ❌ val 与 test 名单存在重叠标本 {len(overlap_spec)} 条，样例：{list(overlap_spec)[:10]}")
        raise AssertionError("val/test specimen sets overlap.")

    # === 2) 类别映射 ===
    base_ds = getattr(dataset, "dataset", dataset)
    class_to_idx = getattr(base_ds, "class_to_idx", None)
    idx_to_class = {v: k for k, v in class_to_idx.items()} if isinstance(class_to_idx, dict) else None

    # === 3) 遍历样本，按优先级分配到 val / test / train ===
    val_idx, test_idx, train_idx = [], [], []
    specimen_hits_val = {}
    specimen_hits_test = {}
    mismatch_warn = 0

    for i in range(len(dataset)):
        di = dataset[i]
        img_path = getattr(di, "img_path", "")

        # 先匹配 val
        hit_val, spn_val = path_contains_any_specimen(img_path, val_specimens)
        # 再匹配 test
        hit_test, spn_test = path_contains_any_specimen(img_path, test_specimens) if test_specimens else (False, "")

        # 双命中（理论不应发生，前面已检查名单不重叠；若发生，取 val 优先并告警）
        if hit_val and hit_test:
            logger.warning(f"[Split] 同时命中 val/test（优先归入 val）：{img_path} | val={spn_val} | test={spn_test}")

        if hit_val:
            val_idx.append(i)
            specimen_hits_val.setdefault(spn_val, []).append(i)
            if idx_to_class is not None:
                ds_cls_name = idx_to_class.get(int(di.y.item()), str(int(di.y.item())))
                if spn_val in val_specimen_to_cls:
                    true_cls_name = val_specimen_to_cls[spn_val]
                    if true_cls_name and (true_cls_name != ds_cls_name):
                        mismatch_warn += 1
                        logger.warning(f"[Split] (val) 标本 '{spn_val}' 分类不一致：list='{true_cls_name}', "
                                       f"dataset='{ds_cls_name}' | {img_path}")
        elif hit_test:
            test_idx.append(i)
            specimen_hits_test.setdefault(spn_test, []).append(i)
            if idx_to_class is not None:
                ds_cls_name = idx_to_class.get(int(di.y.item()), str(int(di.y.item())))
                if spn_test in test_specimen_to_cls:
                    true_cls_name = test_specimen_to_cls[spn_test]
                    if true_cls_name and (true_cls_name != ds_cls_name):
                        mismatch_warn += 1
                        logger.warning(f"[Split] (test) 标本 '{spn_test}' 分类不一致：list='{true_cls_name}', "
                                       f"dataset='{ds_cls_name}' | {img_path}")
        else:
            train_idx.append(i)

    logger.info(f"[Split] 固定划分完成：train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # === 4) 未命中的标本提醒 ===
    unmatched_val = sorted([spn for spn in val_specimens if spn not in specimen_hits_val])
    if unmatched_val:
        logger.warning(f"[Split] (val) 未命中的标本 {len(unmatched_val)} 条（前20）：{unmatched_val[:20]}")

    if test_specimens:
        unmatched_test = sorted([spn for spn in test_specimens if spn not in specimen_hits_test])
        if unmatched_test:
            logger.warning(f"[Split] (test) 未命中的标本 {len(unmatched_test)} 条（前20）：{unmatched_test[:20]}")

    # === 5) 分布表 ===
    train_rows, _ = count_by_class(train_idx, dataset, idx_to_class)
    val_rows, _   = count_by_class(val_idx, dataset, idx_to_class)
    logger.info(pretty_table(train_rows, "[Split] 训练集类别分布"))
    logger.info(pretty_table(val_rows,   "[Split] 验证集类别分布"))
    if test_idx:
        test_rows, _ = count_by_class(test_idx, dataset, idx_to_class)
        logger.info(pretty_table(test_rows, "[Split] 测试集类别分布"))

    # === 6) 互斥与覆盖检查 ===
    assert not set(train_idx).intersection(val_idx), "Train/Val indices overlap!"
    assert not set(train_idx).intersection(test_idx), "Train/Test indices overlap!"
    assert not set(val_idx).intersection(test_idx),   "Val/Test indices overlap!"

    all_idx_sorted = sorted(train_idx + val_idx + test_idx)
    assert all_idx_sorted == list(range(len(dataset))), "Train+Val+Test 未覆盖全部样本或有重复。"

    # === 7) 落盘 ===
    split_dir = os.path.join(save_root, "split_dump")
    os.makedirs(split_dir, exist_ok=True)

    def _dump_indices(fname, idx_list):
        with open(os.path.join(split_dir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, idx_list)))

    _dump_indices("train_indices.txt", train_idx)
    _dump_indices("val_indices.txt",   val_idx)
    _dump_indices("test_indices.txt",  test_idx)  # 即使空也写出，便于后续统一读取

    # manifest（val/test）
    def _dump_manifest(fname, idx_list, specimen_map):
        path = os.path.join(split_dir, fname)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "class", "img_path", "matched_specimen"])
            for i in idx_list:
                di = dataset[i]
                img_path = getattr(di, "img_path", "")
                # 根据对应名单再匹配一次，拿到命中标本号
                hit, spn = path_contains_any_specimen(img_path, set(specimen_map.keys()))
                cname = (idx_to_class.get(int(di.y.item()), str(int(di.y.item())))
                         if idx_to_class else str(int(di.y.item())))
                w.writerow([i, cname, img_path, spn])

    _dump_manifest("val_manifest.csv",  val_idx,  val_specimen_to_cls)
    if test_idx:
        _dump_manifest("test_manifest.csv", test_idx, test_specimen_to_cls)

    logger.info(f"[Split] 划分明细已导出：{split_dir}")

    return train_idx, val_idx, test_idx, idx_to_class

import os
from torch.utils.data import Subset

from typing import Optional, Tuple

def make_fusion_datasets(
    pyg_train,
    pyg_val,
    cfg,
    pyg_test: Optional[object] = None,
    *,
    logger=None,
):
    """
    基于已分好索引得到的 PyG 子集（pyg_train / pyg_val / 可选 pyg_test），
    构建三路融合数据集并返回 (ds_train, ds_val, ds_test)。

    约定：
    - 训练集：rf_pick_mode="random_one"
    - 验证/测试：rf_pick_mode="logit_mean"
    - 若未传 pyg_test，则 ds_test=None
    """
    # === 训练集 ===
    ds_train = FusionDataset(
        pyg_train,
        image_size=cfg.image_size,
        feat_stride=cfg.feat_stride,
        use_imagenet_norm=True,
        rf_csv=cfg.rf_csv,
        num_classes=cfg.rf_num_classes,
        rf_key_col="specimen_id",
        rf_pick_mode="random_one",   # 训练随机挑一条
        rf_random_seed=cfg.split_seed,
        rf_verbose=True,
        morph_area_idx=0,
        morph_perim_idx=1,
        radius_alpha=1.0,
        min_radius_px=2.0,
        max_radius_px=32.0,
        pos_format=cfg.pos_format,
        verbose=True,
    )

    # === 验证集 ===
    ds_val = FusionDataset(
        pyg_val,
        image_size=cfg.image_size,
        feat_stride=cfg.feat_stride,
        use_imagenet_norm=True,
        rf_csv=cfg.rf_csv,
        num_classes=cfg.rf_num_classes,
        rf_key_col="specimen_id",
        rf_pick_mode="logit_mean",   # 验证稳：同一 specimen 多行做 logit-mean
        rf_random_seed=cfg.split_seed,
        rf_verbose=False,
        morph_area_idx=0,
        morph_perim_idx=1,
        radius_alpha=1.0,
        min_radius_px=2.0,
        max_radius_px=32.0,
        pos_format=cfg.pos_format,
        verbose=False,
    )

    # === 测试集（若提供）===
    ds_test = None
    if pyg_test is not None:
        ds_test = FusionDataset(
            pyg_test,
            image_size=cfg.image_size,
            feat_stride=cfg.feat_stride,
            use_imagenet_norm=True,
            rf_csv=cfg.rf_csv,
            num_classes=cfg.rf_num_classes,
            rf_key_col="specimen_id",
            rf_pick_mode="logit_mean",   # 测试与验证一致
            rf_random_seed=cfg.split_seed,
            rf_verbose=False,
            morph_area_idx=0,
            morph_perim_idx=1,
            radius_alpha=1.0,
            min_radius_px=2.0,
            max_radius_px=32.0,
            pos_format=cfg.pos_format,
            verbose=False,
        )
        if logger:
            logger.info(f"[Split] 测试集构建完成：{len(pyg_test)} 样本")

    return ds_train, ds_val, ds_test
