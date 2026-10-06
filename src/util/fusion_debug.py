from __future__ import annotations
from typing import List, Tuple, Dict, Any, Sequence, Optional
import torch
from torch.utils.data import Subset
# ===== NaN/Inf 守卫 & 钩子（打开/关闭开关） =====
DEBUG_NAN = True   # 需要时改为 False 关闭所有体检

def _iter_tensors(obj):
    if obj is None:
        return
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _iter_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _iter_tensors(x)

def guard_finite(name, *objs, throw=True):
    """检查若干张量/容器是否全是有限数；若发现 NaN/Inf 立即报出来源并终止。"""
    if not DEBUG_NAN:
        return
    for obj in objs:
        for t in _iter_tensors(obj):
            if t.numel() == 0:
                continue
            if not torch.isfinite(t).all():
                bad = t[~torch.isfinite(t)]
                msg = (f"[NaN/Inf] {name}: dtype={t.dtype}, shape={tuple(t.shape)}, "
                       f"min={bad.min().item() if bad.numel()>0 else 'NA'}, "
                       f"max={bad.max().item() if bad.numel()>0 else 'NA'}")
                print(msg, flush=True)
                if throw:
                    raise SystemExit(f"Non-finite at {name}")

def install_nan_hooks(model):
    """给关键子模块挂 forward(pre) 钩子，谁先产出 NaN/Inf 当场终止，打印模块名。"""
    if not DEBUG_NAN:
        return []
    handles = []

    def make_pre(name):
        def _pre(mod, inputs):
            guard_finite(f"{name}::inputs", inputs, throw=True)
        return _pre

    def make_post(name):
        def _post(mod, inputs, output):
            guard_finite(f"{name}::outputs", output, throw=True)
        return _post

    watch = {
        "img_enc": model.img_enc,
        "graph_enc": model.graph_enc,
        "branch_a": model.branch_a,
        "branch_b": model.branch_b,
        "fuse_proj": model.fuse_proj,
        "cls_head": model.cls_head,
        "meta_vote": model.meta_vote,
    }
    for name, m in watch.items():
        handles.append(m.register_forward_pre_hook(make_pre(name)))
        handles.append(m.register_forward_hook(make_post(name)))
    return handles

# 你已有的内容（示意）：
# DEBUG_NAN = True
# def _iter_tensors(...): ...
# def guard_finite(...): ...
# def install_nan_hooks(...): ...

# ---------- 数据清洗工具 ----------

def _bad_cols(t: torch.Tensor) -> List[int]:
    """返回含 NaN/Inf 的列索引（用于 2D 特征，如 x/edge_attr）"""
    if t is None or t.numel() == 0 or t.dim() != 2 or not torch.is_floating_point(t):
        return []
    bad = ~torch.isfinite(t)  # [N,F]
    return bad.any(dim=0).nonzero(as_tuple=False).view(-1).tolist()

def _sanitize_inplace(data, clamp: float = 1e6, fields: Sequence[str] = ("x","edge_attr","pos_xy_norm","node_pos_feat","node_r_feat")) -> Tuple[bool, bool]:
    """
    对 Data 中的若干浮点张量做 nan_to_num + clamp，原地写回。
    返回 (fixed_any, still_bad)：
      - fixed_any: 是否做过修复
      - still_bad: 修后是否仍存在非有限数
    """
    fixed_any = False
    still_bad = False
    for attr in fields:
        if hasattr(data, attr):
            t = getattr(data, attr)
            if t is None or not torch.is_tensor(t) or not torch.is_floating_point(t):
                continue
            if (~torch.isfinite(t)).any():
                t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                t.clamp_(-clamp, clamp)
                fixed_any = True
            if not torch.isfinite(t).all():
                still_bad = True
            setattr(data, attr, t)
    return fixed_any, still_bad

def _data_ident(di) -> Dict[str, Any]:
    sid = getattr(di, "specimen_id", None)
    pth = getattr(di, "img_path", None)
    return {"specimen_id": sid, "img_path": pth}

def clean_graph_dataset(
    dataset,
    logger,
    *,
    mode: str = "strict_drop",
    # strict_drop 时完全复刻你原行为：只检查 x
    fields_strict: Sequence[str] = ("x",),
    # fix_then_drop 时会尝试修复这些字段
    fields_fix: Sequence[str] = ("x","edge_attr","pos_xy_norm","node_pos_feat","node_r_feat"),
    clamp: float = 1e6,
    max_log: int = 10,
    return_indices: bool = False,
) -> Tuple[Subset, Dict[str, int]] | Tuple[Subset, Dict[str, int], List[int]]:
    """
    清洗 PyG 数据集。

    mode:
      - 'strict_drop': 复刻旧逻辑：只看 x，空/Nan/Inf 就丢弃。
      - 'fix_then_drop': 先修复 fields_fix，修不掉的丢弃。

    返回：
      - cleaned_ds: Subset 包装后的数据集
      - stats: 统计字典
      - (可选) kept_indices: 被保留的原索引列表（return_indices=True 时返回）
    """
    assert mode in ("strict_drop", "fix_then_drop")
    valid_indices: List[int] = []
    orig_total = len(dataset)
    stats = {
        "kept": 0,
        "fixed": 0,
        "dropped_empty": 0,
        "dropped_nan": 0,   # 仅 strict_drop 使用（与旧逻辑一致）
        "dropped_inf": 0,   # 仅 strict_drop 使用（与旧逻辑一致）
        "dropped_naninf": 0 # fix_then_drop 使用
    }
    examples_logged = 0

    for i in range(len(dataset)):
        di = dataset[i]

        # 空图（无 x 或 x 为空）
        if (getattr(di, "x", None) is None or
            not torch.is_tensor(di.x) or
            di.x.numel() == 0 or di.x.size(0) == 0):
            stats["dropped_empty"] += 1
            if examples_logged < max_log:
                logger.warning(f"[Clean] empty graph @idx={i} | info={_data_ident(di)}")
                examples_logged += 1
            continue

        if mode == "strict_drop":
            # 只检查 x，碰到 NaN/Inf 直接丢弃（复刻你原始行为）
            x = di.x
            if torch.isnan(x).any():
                stats["dropped_nan"] += 1
                if examples_logged < max_log:
                    logger.error(f"[Clean] drop (NaN in x) @idx={i} | info={_data_ident(di)} "
                                 f"| bad_cols_x={_bad_cols(x)}")
                    examples_logged += 1
                continue
            if torch.isinf(x).any():
                stats["dropped_inf"] += 1
                if examples_logged < max_log:
                    logger.error(f"[Clean] drop (Inf in x) @idx={i} | info={_data_ident(di)} "
                                 f"| bad_cols_x={_bad_cols(x)}")
                    examples_logged += 1
                continue

            # 通过
            valid_indices.append(i)
            stats["kept"] += 1

        else:  # fix_then_drop
            # 先记录是否原本有坏值（仅用于日志）
            has_bad_before = False
            for attr in fields_fix:
                if hasattr(di, attr):
                    t = getattr(di, attr)
                    if torch.is_tensor(t) and torch.is_floating_point(t) and (~torch.isfinite(t)).any():
                        has_bad_before = True
                        break

            # 修复
            fixed, still_bad = _sanitize_inplace(di, clamp=clamp, fields=fields_fix)

            if still_bad:
                stats["dropped_naninf"] += 1
                if examples_logged < max_log:
                    logger.error(f"[Clean] drop (NaN/Inf after fix) @idx={i} | info={_data_ident(di)} "
                                 f"| bad_cols_x={_bad_cols(getattr(di,'x', None))} "
                                 f"| bad_cols_edge_attr={_bad_cols(getattr(di,'edge_attr', None))}")
                    examples_logged += 1
                continue

            if fixed and has_bad_before and examples_logged < max_log:
                logger.warning(f"[Clean] fixed @idx={i} | info={_data_ident(di)} "
                               f"| bad_cols_x={_bad_cols(getattr(di,'x', None))} "
                               f"| bad_cols_edge_attr={_bad_cols(getattr(di,'edge_attr', None))}")
                stats["fixed"] += 1
                examples_logged += 1

            valid_indices.append(i)
            stats["kept"] += 1

    cleaned = Subset(dataset, valid_indices) if len(valid_indices) != orig_total else dataset

    # 汇总日志（保持与你原来的风格一致）
    if mode == "strict_drop":
        logger.warning(
            f"[Clean] 过滤异常图: empty={stats['dropped_empty']}, "
            f"nan={stats['dropped_nan']}, inf={stats['dropped_inf']}. "
            f"保留 {stats['kept']} / {orig_total}"
        )
    else:
        logger.info(
            "[Clean] 完成 | "
            f"总数={orig_total} | 保留={stats['kept']} | 修复后保留={stats['fixed']} | "
            f"丢弃(empty)={stats['dropped_empty']} | 丢弃(NaN/Inf)={stats['dropped_naninf']}"
        )

    return (cleaned, stats, valid_indices) if return_indices else (cleaned, stats)
