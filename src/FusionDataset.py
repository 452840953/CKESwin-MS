# fusion_dataset.py
from typing import Optional, Dict, Any, Tuple, List
import os
import math
import csv
import random

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from PIL import Image
from torchvision.transforms import functional as TF

# === logger ===
from util.logger_utils import setup_logger
logger = setup_logger(log_dir="logs", log_prefix="create_fusion_dataset")

def resolve_image_path(value):
    from pathlib import Path
    from create_graph.config import IMAGEROOT
    normalized = str(value).replace("\\", "/")
    if Path(normalized).is_file():
        return normalized
    old_root = os.getenv("CKESWIN_ORIGINAL_IMAGE_ROOT", "").replace("\\", "/").rstrip("/")
    marker = "dataset/images/random_test/"
    if old_root and normalized.startswith(old_root + "/"):
        relative = normalized[len(old_root) + 1:]
    elif marker in normalized:
        relative = normalized.split(marker, 1)[1]
    elif not Path(normalized).is_absolute() and ":" not in normalized:
        relative = normalized
    else:
        raise FileNotFoundError("Cannot remap image path; set CKESWIN_ORIGINAL_IMAGE_ROOT and IMAGEROOT.")
    resolved = Path(IMAGEROOT) / relative
    if not resolved.is_file():
        raise FileNotFoundError(f"Image not found: {resolved}")
    return str(resolved)


# ------------------------- 小工具：张量统计 -------------------------
def _tensor_stats(t: torch.Tensor, name: str) -> str:
    if t is None or t.numel() == 0:
        return f"{name}: EMPTY"
    t_f = t.detach().float()
    return (f"{name}: shape={tuple(t_f.shape)}, "
            f"min={float(t_f.min()):.4g}, max={float(t_f.max()):.4g}, "
            f"mean={float(t_f.mean()):.4g}")

# ------------------------- 坐标&半径工具 -------------------------
def _to_resized_coords(pos_px: torch.Tensor, sy: float, sx: float) -> torch.Tensor:
    """原图像素坐标 -> resize 后像素坐标 (y', x')."""
    if pos_px.numel() == 0:
        return pos_px
    out = pos_px.clone()
    out[:, 0] = out[:, 0] * sy
    out[:, 1] = out[:, 1] * sx
    return out

def _to_feat_coords(pos_resized: torch.Tensor, stride: int) -> torch.Tensor:
    """resize 后像素坐标 -> 特征图坐标 (除以 stride)。"""
    if pos_resized.numel() == 0:
        return pos_resized
    return pos_resized / float(stride)

def _as_pixel_coords(pos_yx: torch.Tensor, H0: int, W0: int) -> torch.Tensor:
    """夹到原图范围内的像素坐标 (y,x)。"""
    if pos_yx.numel() == 0:
        return pos_yx
    pos = pos_yx.clone().float()
    pos[:, 0] = pos[:, 0].clamp(0, max(0, H0 - 1))
    pos[:, 1] = pos[:, 1].clamp(0, max(0, W0 - 1))
    return pos

def _estimate_radius_from_morph(
    x_node: torch.Tensor,
    idx_area: Optional[int] = None,
    idx_perim: Optional[int] = None,
    alpha: float = 1.0,
    min_r: float = 2.0,
    max_r: float = 32.0,
) -> torch.Tensor:
    N = int(x_node.size(0)) if x_node is not None else 0
    if N == 0:
        logger.debug("[radius] x_node is empty -> return empty radii.")
        return torch.empty((0, 1), dtype=torch.float32)

    # 先给出 [N,1] 的默认值
    r = torch.full((N, 1), float(min_r), dtype=torch.float32)

    if x_node is not None and x_node.numel() > 0:
        if idx_area is not None and 0 <= idx_area < x_node.size(1):
            area = x_node[:, idx_area].clamp_min(1e-6).float()            # [N]
            r = (alpha * torch.sqrt(area / math.pi)).unsqueeze(1)         # [N,1] ✅
            logger.debug("[radius] used AREA column for radius.")
        elif idx_perim is not None and 0 <= idx_perim < x_node.size(1):
            perim = x_node[:, idx_perim].clamp_min(1e-6).float()          # [N]
            r = (alpha * (perim / (2.0 * math.pi))).unsqueeze(1)          # [N,1] ✅
            logger.debug("[radius] used PERIMETER column for radius.")
        else:
            logger.debug("[radius] no area/perim provided -> use constant min_r.")

    r = r.clamp(min=min_r, max=max_r)
    return r  # [N,1]


# ------------------------- RF 查表（按“包含关系”匹配） -------------------------
def _safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # 先裁到非负并按行归一化，再加 eps 取 log -> 更稳
    x = x.clamp_min(0)
    row_sum = x.sum(dim=1, keepdim=True).clamp_min(eps)
    x = x / row_sum
    return (x + eps).log()

class RFCsvLookupContains:
    """
    从 CSV 读取多行质谱预测，并在查询时：
      - 给定 img_path
      - 找出所有满足 (key in img_path) 的行（大小写不敏感）
      - 根据 mode 返回单个 [C] 概率向量
    CSV 至少包含: key_col + p0..p{C-1}
    """
    def __init__(self, csv_path: Optional[str], num_classes: Optional[int],
                 key_col: str = "specimen_id", seed: int = 2025, verbose: bool = True):
        self.num_classes = num_classes
        self.key_col = key_col
        self.rng = random.Random(seed)
        self.verbose = verbose

        # key -> List[[C]] 可能同一个 key 有多行
        self.table: Dict[str, List[torch.Tensor]] = {}
        # 另存 keys 的“大小写折叠”版本，用于包含匹配
        self._keys_casefold: List[str] = []

        if not csv_path or not num_classes:
            logger.info("[RF] csv_path or num_classes not set -> RF disabled.")
            return
        if not os.path.exists(csv_path):
            logger.warning(f"[RF] CSV not found: {csv_path} -> RF disabled.")
            return

        logger.info(f"[RF] Loading CSV: {csv_path} | num_classes={num_classes} | key_col='{key_col}'")
        rows = 0
        kept = 0
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows += 1
                if self.key_col not in row:
                    logger.debug(f"[RF] skip row (missing key_col): {row}")
                    continue
                key_raw = str(row[self.key_col]).strip()
                if key_raw == "":
                    logger.debug(f"[RF] skip row (empty key): {row}")
                    continue

                vec = []
                ok = True
                for i in range(num_classes):
                    col = f"p{i}"
                    if col not in row:
                        ok = False
                        break
                    try:
                        vec.append(float(row[col]))
                    except Exception:
                        ok = False
                        break
                if not ok:
                    logger.debug(f"[RF] skip row (bad prob cols): key={key_raw} row={row}")
                    continue

                self.table.setdefault(key_raw, []).append(torch.tensor(vec, dtype=torch.float32))
                kept += 1

        self._keys_casefold = [k.casefold() for k in self.table.keys()]

        logger.info(f"[RF] Loaded rows={rows}, kept={kept}, unique_keys={len(self.table)}")
        if self.verbose and len(self.table) > 0:
            # 打印前几个 key 的样本数量
            preview = list(self.table.items())[:5]
            for k, lst in preview:
                logger.info(f"[RF] key='{k}' -> {len(lst)} samples")

    def get_by_path(self, img_path: str, mode: str = "random_one") -> Optional[torch.Tensor]:
        """
        在 CSV 的 key 上做“包含匹配”： key.casefold() in img_path.casefold()
        收集所有匹配行，然后：
          - random_one : 随机挑 1 行
          - mean       : 算术均值
          - logit_mean : 先取 log 再均值再 softmax（稳一点）
        """
        if not self.table:
            return None

        ip = img_path.casefold()
        candidate_vecs: List[torch.Tensor] = []
        for key_raw, key_cf in zip(self.table.keys(), self._keys_casefold):
            if key_cf in ip:
                candidate_vecs.extend(self.table[key_raw])

        if not candidate_vecs:
            logger.warning(f"[RF] No RF match for img_path contains any key: {img_path}")
            return None

        if mode == "random_one":
            vec = candidate_vecs[self.rng.randrange(len(candidate_vecs))]
            logger.debug(f"[RF] pick=random_one, candidates={len(candidate_vecs)}")
            return vec

        V = torch.stack(candidate_vecs, dim=0)  # [K,C]
        if mode == "mean":
            vec = V.mean(dim=0)
            logger.debug(f"[RF] pick=mean, candidates={len(candidate_vecs)} | {_tensor_stats(vec, 'rf_mean')}")
            return vec
        if mode == "logit_mean":
            m = _safe_log(V).mean(dim=0)
            vec = torch.softmax(m, dim=0)
            logger.debug(f"[RF] pick=logit_mean, candidates={len(candidate_vecs)} | {_tensor_stats(vec, 'rf_logit_mean')}")
            return vec

        # 兜底
        vec = candidate_vecs[self.rng.randrange(len(candidate_vecs))]
        logger.debug(f"[RF] pick=fallback(random_one), candidates={len(candidate_vecs)}")
        return vec


# ------------------------- 主数据集：图+图像(+RF) -------------------------
class FusionDataset(Dataset):
    """
    使 WoodCellsGraphDataset 产出的 PyG Data（含 .pos, .x, .img_path）与图像对齐，
    并附加节点对齐坐标/半径；质谱部分：
      - 直接用 **整条 img_path** 去 CSV 里做“包含匹配”
      - 若匹配到多行：训练随机抽 1 条（rf_pick_mode="random_one"）
      - 验证/测试可用 "mean" / "logit_mean" 聚合

    输出 sample:
        {
            "graph": Data(... with node_pos_px/node_pos_feat/...),
            "image": Tensor [3,Ht,Wt],
            "rf_proba": Tensor [C] or None,
            "meta": {...}
        }
    """
    def __init__(
        self,
        pyg_dataset,                           # WoodCellsGraphDataset 或其 Subset
        image_size: Tuple[int, int] = (2048, 2048),
        feat_stride: int = 16,                 # CNN/ViT 取用特征图的步幅
        use_imagenet_norm: bool = True,        # 若用 ImageNet 预训练骨干建议 True
        norm_mean: Tuple[float,float,float] = (0.485, 0.456, 0.406),
        norm_std: Tuple[float,float,float]   = (0.229, 0.224, 0.225),

        # --- RF 相关（包含匹配） ---
        rf_csv: Optional[str] = None,
        num_classes: Optional[int] = None,
        rf_key_col: str = "specimen_id",       # CSV 中 key 列名（用其值去做“包含匹配”）
        rf_pick_mode: str = "random_one",      # 训练: random_one；验证/测试: logit_mean/mean
        rf_random_seed: int = 2025,            # 控制随机挑选的种子
        rf_verbose: bool = True,

        # --- 半径估计 ---
        morph_area_idx: Optional[int] = None,  # x 中“面积”列索引
        morph_perim_idx: Optional[int] = None, # x 中“周长”列索引
        radius_alpha: float = 1.0,
        min_radius_px: float = 2.0,
        max_radius_px: float = 32.0,

        # --- 你建图的 pos 存储格式 ---
        pos_format: str = "xy",                # 你的建图常用 (x,y)，这里默认 "xy"

        # --- 日志 ---
        verbose: bool = True,                  # True: 输出 info 级别汇总；详细细节在 debug
    ):
        self.base = pyg_dataset
        self.Ht, self.Wt = int(image_size[0]), int(image_size[1])
        self.feat_stride = int(feat_stride)

        self.use_imagenet_norm = bool(use_imagenet_norm)
        self.norm_mean, self.norm_std = norm_mean, norm_std

        # RF 查表器（包含匹配）
        self.rf_lookup = RFCsvLookupContains(
            rf_csv, num_classes, key_col=rf_key_col, seed=rf_random_seed, verbose=rf_verbose
        ) if (rf_csv and num_classes) else None
        self.rf_pick_mode = rf_pick_mode

        # 半径估计
        self.morph_area_idx = morph_area_idx
        self.morph_perim_idx = morph_perim_idx
        self.radius_alpha = float(radius_alpha)
        self.min_radius_px = float(min_radius_px)
        self.max_radius_px = float(max_radius_px)

        assert pos_format in ("xy", "yx")
        self.pos_format = pos_format

        self.verbose = verbose
        if self.verbose:
            logger.info("=== FusionDataset init ===")
            logger.info(f"dataset_len={len(self.base)} | image_size={self.Ht}x{self.Wt} | feat_stride={self.feat_stride}")
            logger.info(f"use_imagenet_norm={self.use_imagenet_norm} | pos_format={self.pos_format}")
            logger.info(f"radius: alpha={self.radius_alpha}, min={self.min_radius_px}, max={self.max_radius_px}, "
                        f"area_idx={self.morph_area_idx}, perim_idx={self.morph_perim_idx}")
            if self.rf_lookup is None:
                logger.info(f"RF: disabled (csv={rf_csv}, num_classes={num_classes})")
            else:
                logger.info(f"RF: enabled, mode={self.rf_pick_mode}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d: Data = self.base[idx]
        if not hasattr(d, "img_path"):
            logger.error(f"[sample {idx}] Data missing img_path!")
            raise KeyError("Data 缺少 img_path，无法对齐图像")
        if not hasattr(d, "pos"):
            logger.error(f"[sample {idx}] Data missing pos!")
            raise KeyError("Data 缺少 pos（节点坐标）")

        # Remap an archived graph's image root without embedding a machine path.
        img_path = resolve_image_path(d.img_path)
        d.img_path = img_path

        # 2) 读图 -> resize -> ToTensor -> (可选) ImageNet 归一化
        try:
            with Image.open(d.img_path).convert("RGB") as img:
                H0, W0 = img.size[1], img.size[0]                # PIL: (W,H)
                sy, sx = self.Ht / H0, self.Wt / W0
                img_r = img.resize((self.Wt, self.Ht), resample=Image.BILINEAR)
        except Exception as e:
            logger.error(f"[sample {idx}] open/resize image failed: {d.img_path} | {repr(e)}")
            raise

        img_t = TF.to_tensor(img_r)                              # [3,Ht,Wt] in [0,1]
        if self.use_imagenet_norm:
            img_t = TF.normalize(img_t, self.norm_mean, self.norm_std)

        if self.verbose and idx < 3:  # 前几个样本打印关键信息
            logger.info(f"[sample {idx}] img='{d.img_path}' | orig=({H0},{W0}) -> resized=({self.Ht},{self.Wt}) "
                        f"| scale=(sy={sy:.4g}, sx={sx:.4g})")

        # 2) 坐标：你的建图 pos 通常是 (x,y) 像素；这里统一为 (y,x)
        pos_xy = d.pos.float()
        pos_yx = pos_xy[:, [1, 0]] if (pos_xy.numel() > 0 and self.pos_format == "xy") else pos_xy
        pos_px0 = _as_pixel_coords(pos_yx, H0, W0)               # 原图像素 (y,x)
        pos_px  = _to_resized_coords(pos_px0, sy, sx)            # resize 后像素 (y',x')
        pos_f   = _to_feat_coords(pos_px, self.feat_stride)      # 特征图坐标

        logger.debug(f"[sample {idx}] {_tensor_stats(pos_px0, 'pos_px0')}")
        logger.debug(f"[sample {idx}] {_tensor_stats(pos_px,  'pos_px')}")
        logger.debug(f"[sample {idx}] {_tensor_stats(pos_f,   'pos_feat')}")

        # 3) 柔性半径：原图像素估计 -> 映射到 resize/特征图
        x_node = d.x.float() if hasattr(d, "x") and d.x is not None else torch.empty((0, 0))
        r_px_orig = _estimate_radius_from_morph(
            x_node,
            idx_area=self.morph_area_idx,
            idx_perim=self.morph_perim_idx,
            alpha=self.radius_alpha,
            min_r=self.min_radius_px,
            max_r=self.max_radius_px,
        )  # (N,1) 原图像素
        ry_px = r_px_orig * sy
        rx_px = r_px_orig * sx
        r_iso_px   = 0.5 * (ry_px + rx_px)                       # (N,1) 等向（resize 像素）
        r_iso_feat = r_iso_px / float(self.feat_stride)
        r_aniso_px   = torch.cat([ry_px, rx_px], dim=1) if r_px_orig.numel() > 0 else torch.empty((0, 2))
        r_aniso_feat = r_aniso_px / float(self.feat_stride) if r_aniso_px.numel() > 0 else r_aniso_px

        logger.debug(f"[sample {idx}] {_tensor_stats(r_px_orig,    'r_px_orig')}")
        logger.debug(f"[sample {idx}] {_tensor_stats(r_iso_px,     'r_iso_px')}")
        logger.debug(f"[sample {idx}] {_tensor_stats(r_iso_feat,   'r_iso_feat')}")
        logger.debug(f"[sample {idx}] {_tensor_stats(r_aniso_px,   'r_aniso_px')}")
        logger.debug(f"[sample {idx}] {_tensor_stats(r_aniso_feat, 'r_aniso_feat')}")

        # 4) RF 概率向量（用 img_path 做“包含匹配”）
        rf_vec = None
        if self.rf_lookup is not None:
            rf_vec = self.rf_lookup.get_by_path(d.img_path, mode=self.rf_pick_mode)  # -> [C] or None
            if rf_vec is None:
                logger.warning(f"[sample {idx}] No RF vector found by contains-match: {d.img_path}")
            else:
                logger.debug(f"[sample {idx}] {_tensor_stats(rf_vec, 'rf_vec')}")

        # 5) 回写到 Data，保证 Batch.from_data_list 能对齐拼接
        d = d.clone()
        d.node_pos_px_orig  = pos_px0
        d.node_pos_px       = pos_px
        d.node_pos_feat     = pos_f
        d.node_r_px         = r_iso_px
        d.node_r_feat       = r_iso_feat
        d.node_r_px_aniso   = r_aniso_px
        d.node_r_feat_aniso = r_aniso_feat
        d.img_wh0           = torch.tensor([H0, W0], dtype=torch.float32)
        d.resize_wh         = torch.tensor([self.Ht, self.Wt], dtype=torch.float32)
        d.feat_stride       = torch.tensor([self.feat_stride], dtype=torch.int64)

        sample: Dict[str, Any] = {
            "graph": d,
            "image": img_t,                # [3,Ht,Wt]
            "rf_proba": rf_vec,            # [C] or None
            "meta": {
                "index": idx,
                "img_path": d.img_path,
                "orig_hw": (H0, W0),
                "scale": (sy, sx),
            },
        }
        return sample

# ------------------------- collate -------------------------
def fusion_collate(samples: List[Dict[str, Any]], showshap=None):
    """
    批量合并：
      - 图：PyG Batch.from_data_list
      - 图像：stack -> [B,3,H,W]
      - RF：若任一为 None 则返回 None，否则 stack -> [B,C]
      - meta：list 保留
    返回： (batch_graph, batch_image, batch_rf, meta_list)
    """
    if len(samples) == 0:
        logger.warning("[collate] empty batch!")
        return None, None, None, []

    graphs = [s["graph"] for s in samples]
    imgs   = [s["image"] for s in samples]
    rf_vec = [s["rf_proba"] for s in samples]
    meta   = [s["meta"] for s in samples]

    batch_graph = Batch.from_data_list(graphs)
    batch_img   = torch.stack(imgs, dim=0)  # [B,3,H,W]

    if any(v is None for v in rf_vec):
        batch_rf = None
        # 不再每步打 info，如需排查时手动开 debug
        logger.debug(f"[collate] B={len(samples)} | img={tuple(batch_img.shape)} | rf=None (some missing)")
    else:
        batch_rf = torch.stack(rf_vec, dim=0)  # [B,C]
        logger.debug(f"[collate] B={len(samples)} | img={tuple(batch_img.shape)} | rf={tuple(batch_rf.shape)}")
    
    # 一些关键字段统计（仅 debug 打印，避免每步刷屏）
    logger.debug(_tensor_stats(batch_graph.node_pos_px, "batch.node_pos_px"))
    logger.debug(_tensor_stats(batch_graph.node_r_px,   "batch.node_r_px"))
    logger.debug(_tensor_stats(batch_graph.node_pos_feat, "batch.node_pos_feat"))
    logger.debug(_tensor_stats(batch_graph.node_r_feat,   "batch.node_r_feat"))

    return batch_graph, batch_img, batch_rf, meta
