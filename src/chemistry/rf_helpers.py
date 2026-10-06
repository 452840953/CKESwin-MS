"""RF helpers extracted from the archived mass-spectrometry training code."""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
FIXED_EVAL_LIST = "inputs/splits/test.txt"

def assert_no_leak(groups_train, groups_eval, inner_cv, X_train, y_train):
    g_tr = set(np.unique(groups_train))
    g_ev = set(np.unique(groups_eval))
    inter = g_tr & g_ev
    if inter:
        raise RuntimeError(f"[泄露] 训练池与eval的组有重叠: {sorted(list(inter))[:10]} ...")

    # 检查前几折
    from itertools import islice
    for i, (tr_idx, va_idx) in enumerate(islice(inner_cv.split(X_train, y_train, groups=groups_train), 5)):
        gt = set(np.unique(groups_train[tr_idx]))
        gv = set(np.unique(groups_train[va_idx]))
        if gt & gv:
            raise RuntimeError(f"[泄露] CV第{i}折 训练/验证 组有重叠")
    print("[检查] 组重叠检查通过：训练池 vs eval 互斥；CV 每折 train/val 组互斥。")

def read_csv_auto_encoding(path: str, header=None) -> pd.DataFrame:
    """
    自动尝试几种常见编码读取 CSV（先 UTF-8-SIG，再 GBK，再 latin1）。
    """
    for enc in ("utf-8-sig", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, header=header)
        except Exception:
            continue
    # 最后尝试默认
    return pd.read_csv(path, header=header)

def ensure_numeric_df(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    将全部特征列转为数值；非数值强制为 NaN，再用列中位数填充；全 NaN 列用 0。
    """
    out = pd.DataFrame()
    for c in df_feat.columns:
        col = pd.to_numeric(df_feat[c], errors="coerce")
        if col.isna().all():
            col = col.fillna(0.0)
        else:
            med = col.median()
            col = col.fillna(med)
        out[c] = col.astype(np.float32)
    return out

def _save_txt(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(str(x) + "\n")

def summarize_split(y, groups, label_decoder=None, name="SPLIT", out_dir=None, indices=None):
    """
    打印并可选保存：样本数、组数、类别-样本数、类别-组数清单。
    indices: 若提供，仅统计该子集；否则统计全部。
    """
    if indices is not None:
        y = np.asarray(y)[indices]
        groups = np.asarray(groups)[indices]

    n_samples = len(y)
    uniq_groups, grp_counts = np.unique(groups, return_counts=True)
    n_groups = len(uniq_groups)
    print(f"[{name}] 组数: {n_groups}，样本数: {n_samples}")

    classes, counts = np.unique(y, return_counts=True)
    lines_summary = []
    for c, cnt in zip(classes, counts):
        gset = np.unique(groups[y == c])
        label_name = label_decoder[c] if (label_decoder is not None and c < len(label_decoder)) else c
        print(f"  - 类别 {c} ({label_name}) -> 样本 {cnt}，组数 {len(gset)}，组: {list(gset)}")
        lines_summary.append(f"class {c} ({label_name}) | n={cnt}, groups={len(gset)} | {list(gset)}")

    if out_dir is not None:
        # 保存“组-样本量”清单
        df_grp = pd.DataFrame({"group": uniq_groups, "count": grp_counts}).sort_values("group")
        save_df(df_grp, os.path.join(out_dir, f"{name.lower()}_groups_counts.csv"))
        # 保存类别-组摘要
        _save_txt(lines_summary, os.path.join(out_dir, f"{name.lower()}_class_group_summary.txt"))

def _read_fixed_eval_groups(txt_path: str) -> set:
    """
    从固定 txt 里读取 eval 的“分组键”（即 CSV 的第2列；txt 每行形如 'W22639,Pterocarpus_xxx'）。
    仅取逗号前的部分，并去除空白。
    """
    eval_groups = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 允许存在多余空格
            if "," in line:
                g = line.split(",", 1)[0].strip()
            else:
                g = line.strip()
            if g:
                eval_groups.add(g)
    return eval_groups

def train_eval_split_fixed_by_groups(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    fixed_eval_groups: set,
    inner_train_ratio=0.75,
    random_state=42,
    print_info=True,
    out_dir=None,             # 新增：写文件留痕
    label_decoder=None        # 新增：打印原始标签名
):
    """
    先“按组剥离”固定的 eval（来自 txt），其余作为可训练池。
    返回：
      - X_train, y_train, groups_train（交叉验证只在这个集合上进行）
      - X_eval,  y_eval  （外部评估，固定不变）
      - inner_cv_splitter（GroupShuffleSplit，train_size=0.75, test_size=0.25）
    """
    groups = np.asarray(groups).astype(str)
    y = np.asarray(y)

    # 哪些固定eval组没有出现在当前CSV（提醒一下）
    missing = sorted(list(fixed_eval_groups - set(groups)))
    if print_info and missing:
        print(f"[固定EVAL] 警告：txt中有 {len(missing)} 个分组未在该CSV出现：{missing[:10]}{' ...' if len(missing)>10 else ''}")

    is_eval = np.array([g in fixed_eval_groups for g in groups], dtype=bool)
    eval_idx = np.where(is_eval)[0]
    train_pool_idx = np.where(~is_eval)[0]

    if print_info:
        print(f"[固定EVAL] 从 {FIXED_EVAL_LIST} 读取到 {len(fixed_eval_groups)} 个分组键。")
        if len(fixed_eval_groups) > 0:
            ex = list(sorted(fixed_eval_groups))[:12]
            print(f"[固定EVAL] txt样例: {ex}")
        print(f"[固定EVAL] 在当前CSV命中 组数: {len(np.unique(groups[eval_idx]))}，样本: {len(eval_idx)}")
        print(f"[固定EVAL] 训练池样本: {len(train_pool_idx)}")

        n_all = len(groups)
        prop_eval = len(eval_idx) / max(1, n_all)
        print(f"[固定EVAL] 全局占比 eval≈{prop_eval:.3f}（目标参考 0.20） | 全局样本={n_all}, eval={len(eval_idx)}, 余量={len(train_pool_idx)}")

    # 切出 eval
    X_eval, y_eval = X[eval_idx], y[eval_idx]

    # 剩下的是“训练池”（用于寻参CV）
    X_train, y_train = X[train_pool_idx], y[train_pool_idx]
    groups_train = groups[train_pool_idx]

    # === 详细汇总（屏幕 + 文件）===
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        _save_txt(sorted(list(np.unique(groups[eval_idx]))), os.path.join(out_dir, "eval_groups.txt"))
        _save_txt(sorted(list(np.unique(groups_train))), os.path.join(out_dir, "trainpool_groups.txt"))
        # 保存索引
        save_json({"eval_idx": eval_idx.tolist(), "train_pool_idx": train_pool_idx.tolist()},
                  os.path.join(out_dir, "split_indices_fixed_eval.json"))

    if print_info:
        summarize_split(y, groups, label_decoder=label_decoder, name="EVAL", out_dir=out_dir, indices=eval_idx)
        summarize_split(y, groups, label_decoder=label_decoder, name="TRAINPOOL", out_dir=out_dir, indices=train_pool_idx)

    # === 构造组保持的CV（0.75:0.25）===
    from sklearn.model_selection import GroupShuffleSplit
    inner_cv_splitter = GroupShuffleSplit(
        n_splits=5, train_size=inner_train_ratio, test_size=1 - inner_train_ratio, random_state=random_state
    )

    # 小预览：用第1个split打印一次“训练子集 / 验证子集”规模与类别分布
    # === 构造组保持的CV（0.75:0.25）===
    from sklearn.model_selection import GroupShuffleSplit
    inner_cv_splitter = GroupShuffleSplit(
        n_splits=5, train_size=inner_train_ratio, test_size=1 - inner_train_ratio, random_state=random_state
    )

    # 打印并保存所有折的划分
    if print_info:
        try:
            for k, (tr_sub, va_sub) in enumerate(
                inner_cv_splitter.split(X_train, y_train, groups=groups_train), start=1
            ):
                print(f"[CV第{k}/{inner_cv_splitter.n_splits}] 训练池一次划分 => "
                    f"train_sub={len(tr_sub)}，val_sub={len(va_sub)}（期望0.75/0.25）")

                summarize_split(
                    y_train, groups_train, label_decoder=label_decoder,
                    name=f"CV{k}_TRAIN_SUB", out_dir=out_dir, indices=tr_sub
                )
                summarize_split(
                    y_train, groups_train, label_decoder=label_decoder,
                    name=f"CV{k}_VALID_SUB", out_dir=out_dir, indices=va_sub
                )

                if out_dir is not None:
                    save_json(
                        {"train_idx": tr_sub.tolist(), "valid_idx": va_sub.tolist()},
                        os.path.join(out_dir, f"cv_indices_fold{k}.json")
                    )
        except Exception as e:
            print(f"[CV预览] 打印所有折失败：{e}")


    return X_train, y_train, groups_train, X_eval, y_eval, inner_cv_splitter

def compute_metrics(y_true, y_pred, y_proba=None):
    """
    返回常用指标：accuracy、macro_f1、weighted_f1、macro_auc（若有概率）
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    res = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }
    # AUC（多分类）
    if y_proba is not None:
        try:
            # y_proba: [n_samples, n_classes]
            auc_macro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            res["auc_macro_ovr"] = auc_macro
        except Exception:
            pass
    return res

def _to_builtin(o):
    """把包含 numpy 类型的对象递归转换为纯 Python 可 JSON 序列化的对象。"""
    import numpy as _np
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, _np.generic):
        # numpy 标量 -> Python 标量
        return o.item()
    if isinstance(o, _np.ndarray):
        return [_to_builtin(x) for x in o.tolist()]
    if isinstance(o, dict):
        return { _to_builtin(k): _to_builtin(v) for k, v in o.items() }
    if isinstance(o, (list, tuple, set)):
        return [ _to_builtin(x) for x in o ]
    # 兜底：转字符串（尽量少用）
    try:
        return json.loads(json.dumps(o))
    except Exception:
        return str(o)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(obj), f, ensure_ascii=False, indent=2)

def save_df(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
