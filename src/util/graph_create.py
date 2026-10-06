import numpy as np
import os
from typing import Dict, List, Tuple
import glob
import math
import torch
# util/feature_extract.py
import cv2
from create_graph.graph.visualize import calculate_perimeter
from create_graph.graph.build import build_graph_dcse
from create_graph.graph.prune import prune_edges

IMG_EXTS = (
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
    ".PNG", ".JPG", ".JPEG", ".TIF", ".TIFF", ".BMP"
)

NODE_BASE_DIM = 8  # [area, lumen_area, wall_area, cell_perim, lumen_perim, wall_perim, lumen_circ, wall_lumen_ratio]
NODE_EXTRA_DIM = 1 # node_angle_var
NODE_FEAT_DIM = NODE_BASE_DIM + NODE_EXTRA_DIM  # 9 (ALWAYS)
EDGE_FEAT_DIM = 9  # [dist, norm_dist_img, cos, sin, norm_dist_diam, diff_area, diff_circ, diff_ratio, edge_angle_mean_dev]

# 把各种格式的掩码统一转成 uint8 的二值掩码（0 或 255）
def _to_uint8_mask(m):
    if isinstance(m, dict):
        for k in ("segmentation", "mask", "seg", "m"):
            if k in m:
                m = m[k]
                break
    m = np.asarray(m)
    if m.dtype == bool:
        return m.astype(np.uint8) * 255
    if m.size == 0:
        return np.zeros_like(m, dtype=np.uint8)
    if m.max() <= 1:
        return (m > 0).astype(np.uint8) * 255
    return (m > 127).astype(np.uint8) * 255

# 通用边规范化函数，核心作用是把 输入边列表（格式可能很乱、含自环或非法边）统一转为 标准化的二维 NumPy 数组
def _normalize_edges(
    edges,
    num_nodes: int,
    one_based: bool = True,        # ← 你的流水线建议 True
    verbose: bool = False,
    logger=None
) -> np.ndarray:
    """
    将输入边(可能是 dict/list/tuple)规范到 0-based 的 [E,2] 数组。
    - 不重排/压缩节点编号，只做 1→0 平移与合法性校验。
    - 去自环与越界，去重。
    """
    if edges is None:
        return np.zeros((0, 2), dtype=np.int64)

    key_pairs = [("src","dst"), ("source","target"), ("from","to"), ("u","v"), ("i","j")]
    out = []
    converted = 0
    dropped = 0

    for k, e in enumerate(edges):
        # 解析 u, v
        if isinstance(e, (list, tuple)) and len(e) == 2:
            u, v = int(e[0]), int(e[1])
        elif isinstance(e, dict):
            got = None
            for a, b in key_pairs:
                if a in e and b in e:
                    got = (e[a], e[b]); break
            if got is None:
                for a in ("edge","pair","uv","ij"):
                    if a in e and isinstance(e[a], (list, tuple)) and len(e[a]) == 2:
                        got = (e[a][0], e[a][1]); break
            if got is None:
                dropped += 1
                continue
            u, v = int(got[0]), int(got[1])
        else:
            dropped += 1
            continue

        # 1-based → 0-based（或自动识别）
        if one_based:
            u -= 1; v -= 1
            converted += 1
        else:
            # 自动识别：若不在 0-based 范围，但在 1-based 范围，则转一次
            if not (0 <= u < num_nodes and 0 <= v < num_nodes) and (1 <= u <= num_nodes and 1 <= v <= num_nodes):
                u -= 1; v -= 1
                converted += 1

        # 过滤非法与自环
        if u == v or not (0 <= u < num_nodes and 0 <= v < num_nodes):
            dropped += 1
            continue

        out.append((u, v))

    arr = np.unique(np.asarray(out, dtype=np.int64), axis=0) if out else np.zeros((0, 2), dtype=np.int64)

    if verbose and logger is not None:
        logger.info(f"  - 规范化边: 输入={len(edges)} → 输出={arr.shape[0]}  (converted={converted}, dropped={dropped})")

    return arr

# 数据集索引器,用来找类别子文件夹，读取里面图片，打标签，打类别标签
def list_images_with_labels(root_data: str) -> Tuple[List[str], List[int], Dict[str,int]]:
    if not os.path.isdir(root_data):
        raise FileNotFoundError(f"data 目录不存在: {root_data}")

    classes = sorted([d for d in os.listdir(root_data) if os.path.isdir(os.path.join(root_data, d))])
    if not classes:
        raise RuntimeError(f"data 目录下未发现子目录（类别）: {root_data}")

    class_to_idx = {c: i for i, c in enumerate(classes)}

    image_paths, labels = [], []
    for c in classes:
        cdir = os.path.join(root_data, c)
        for ext in IMG_EXTS:
            for p in glob.glob(os.path.join(cdir, f"*{ext}")):
                image_paths.append(p)
                labels.append(class_to_idx[c])

    if len(image_paths) == 0:
        raise RuntimeError(f"在 {root_data} 下未找到任何图片（扩展名: {IMG_EXTS}）")

    return image_paths, labels, class_to_idx

# 将每个细胞对应的属性转节点特征向量，供应torch使用，转tensor类型
def _build_node_base_features(cell_info: List[List[float]]) -> torch.Tensor:
    """Return (N,F) float32 tensor, where F = number of features (excluding index)."""
    if not cell_info:
        return torch.empty((0, 0), dtype=torch.float32)  # 0 节点时返回空
    
    feats = []
    for row in cell_info:
        feats.append([float(v) for v in row[1:]])  # 跳过第一个 idx，只取后面所有特征
    
    return torch.tensor(feats, dtype=torch.float32)

# 算邻居节点的角度分布差异
def _compute_node_topo_feats(
    pos: torch.Tensor, edge_index: torch.Tensor, hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Return (N, K) topo-aware node features.
    Includes:
      - degree
      - mean_neighbor_dist
      - mean_angle_diff
      - angle_std
      - normalized pos_x, pos_y
      - absolute pos_x, pos_y
    """
    H, W = hw
    N = pos.size(0)
    if N == 0:
        return torch.empty((0, 8), dtype=torch.float32)
    if edge_index.numel() == 0:
        # 只有位置信息，没有拓扑
        pos_norm = torch.stack([pos[:,0] / W, pos[:,1] / H], dim=1)
        return torch.cat([
            torch.zeros((N, 4), dtype=torch.float32),  # 拓扑统计
            pos_norm,                                  # 归一化位置
            pos                                        # 绝对位置
        ], dim=1)

    src, dst = edge_index
    feats = torch.zeros((N, 8), dtype=torch.float32)
    nbr_lists = [[] for _ in range(N)]
    for i, j in zip(src.tolist(), dst.tolist()):
        nbr_lists[i].append(j)

    for i in range(N):
        nbrs = nbr_lists[i]
        if len(nbrs) == 0:
            degree, mean_dist, mean_angle, angle_std = 0.0, 0.0, 0.0, 0.0
        else:
            vecs = pos[nbrs] - pos[i]
            # 1) degree
            degree = len(nbrs)
            # 2) mean neighbor distance
            mean_dist = vecs.norm(dim=1).mean().item()
            # 3) angle features
            if len(nbrs) > 1:
                angles = torch.atan2(vecs[:, 1], vecs[:, 0])  # [k]
                diffs = angles.unsqueeze(0) - angles.unsqueeze(1)
                diffs = torch.remainder(diffs + math.pi, 2 * math.pi) - math.pi
                mean_angle = diffs.abs().mean().item()
                angle_std = diffs.std().item()
            else:
                mean_angle, angle_std = 0.0, 0.0

        # 4) normalized pos
        px_norm, py_norm = pos[i,0].item() / W, pos[i,1].item() / H
        # 5) absolute pos
        px_abs, py_abs   = pos[i,0].item(), pos[i,1].item()

        feats[i] = torch.tensor([
            degree, mean_dist, mean_angle, angle_std,
            px_norm, py_norm,
            px_abs, py_abs
        ])

    return feats


# 把剪枝后的边列表转成 PyTorch
def _build_edge_index(edges_pruned, num_nodes: int, undirected: bool, verbose: bool, logger=None) -> torch.Tensor:
    e_arr = _normalize_edges(edges_pruned, num_nodes, one_based=True, verbose=verbose, logger=logger)
    if e_arr.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if undirected:
        e_rev = np.stack([e_arr[:, 1], e_arr[:, 0]], axis=1)
        e_arr = np.concatenate([e_arr, e_rev], axis=0)
        e_arr = np.unique(e_arr, axis=0)  # 只去重，不改节点编号
    return torch.from_numpy(e_arr.T.copy()).long()


# 边特征构建
def _build_edge_attr(pos: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, hw: Tuple[int,int]) -> torch.Tensor:
    """Return (E, EDGE_FEAT_DIM) tensor; stable for empty edges."""
    IDX_AREA, IDX_PERIM, IDX_CIRC, IDX_AR, IDX_RECT, IDX_SOL, IDX_ECC, IDX_ORI, IDX_COMP = range(9)
    if edge_index.numel() == 0:
        return torch.empty((0, EDGE_FEAT_DIM), dtype=torch.float32)
    # 基本几何量计算
    H, W = hw
    diag_len = float(math.hypot(H, W)) + 1e-6

    src, dst = edge_index             #边起点和终点
    d = pos[src] - pos[dst]           # [E,2]
    # 边在平面上的分量
    dx = d[:, 0].unsqueeze(1)        
    dy = d[:, 1].unsqueeze(1)
    # 欧式距离
    dist = torch.sqrt(dx**2 + dy**2)  # [E,1]
    # 方向角相关量
    norm = dist + 1e-6
    cos_theta = dx / norm             # [E,1]
    sin_theta = dy / norm             # [E,1]
    # 边长归一化到图像尺寸
    norm_dist_img = dist / diag_len   # [E,1]
    # 结合节点的几何属性
    # 两端节点的周长特征
    perim_src = x[src][:, IDX_PERIM]  # 1
    perim_dst = x[dst][:, IDX_PERIM]
    # 两端节点的平均直径
    avg_diam = ((perim_src + perim_dst) / (2*math.pi)).unsqueeze(1) + 1e-6
    # 节点间距除以平均直径，衡量边长相对于细胞大小的比例
    norm_dist_diam = dist / avg_diam
    # 提取节点属性差异
    area_src, area_dst = x[src][:, 0], x[dst][:, 0]
    circ_src, circ_dst = x[src][:, IDX_CIRC], x[dst][:, IDX_CIRC]  # 2
    ratio_src, ratio_dst = x[src][:, IDX_AR], x[dst][:, IDX_AR]
    # 面积差
    # 节点圆度差
    # 长宽比差
    diff_area  = (area_src - area_dst).abs().unsqueeze(1)
    diff_circ  = (circ_src - circ_dst).abs().unsqueeze(1)
    diff_ratio = (ratio_src - ratio_dst).abs().unsqueeze(1)

    # edge-angle mean abs deviation
    edge_angle_stats = []
    # quick adjacency: indices of outgoing neighbors for each src node
    # 邻接表构建
    nbr_lists = [[] for _ in range(pos.size(0))]
    for i, j in zip(src.tolist(), dst.tolist()):
        nbr_lists[i].append(j)
    # 对每条边，计算边的方向角，找出同一起点的其他向量角度，计算偏差，取平均
    pos_np = pos.numpy() if isinstance(pos, torch.Tensor) else pos
    for i, j in zip(src.tolist(), dst.tolist()):
        nbrs = nbr_lists[i]
        if len(nbrs) <= 1:
            edge_angle_stats.append(0.0)
            continue
        dx_ij = pos_np[j, 0] - pos_np[i, 0]
        dy_ij = pos_np[j, 1] - pos_np[i, 1]
        theta_ij = math.atan2(dy_ij, dx_ij)
        vecs = pos[nbrs] - pos[i]
        angles = torch.atan2(vecs[:, 1], vecs[:, 0])
        diffs = (angles - theta_ij + math.pi) % (2 * math.pi) - math.pi
        edge_angle_stats.append(float(diffs.abs().mean().item()))

    edge_angle_stats = torch.tensor(edge_angle_stats, dtype=torch.float32).unsqueeze(1)

    return torch.cat([
        dist,            # 1
        norm_dist_img,   # 1
        cos_theta,       # 1
        sin_theta,       # 1
        norm_dist_diam,  # 1
        diff_area,       # 1
        diff_circ,       # 1
        diff_ratio,      # 1
        edge_angle_stats # 1
    ], dim=1)

def extract_features_from_masks(masks: List[np.ndarray]) -> Tuple[List[Tuple[int, int]], List[List[float]]]:
    """
    批量处理 mask，返回 (centroids, cell_info)
    - centroids: [(cx, cy), ...]
    - cell_info: [
          [idx, area, perimeter, circularity,
           aspect_ratio, rectangularity, solidity,
           eccentricity, orientation_angle, complexity], ...
      ]
    """
    centroids = []
    cell_info = []

    for idx, mask in enumerate(masks):
        if mask.size == 0:
            continue

        # 质心
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        centroids.append((cx, cy))

        # 基础几何特征
        cell_area = int(np.count_nonzero(mask == 255))
        cell_perimeter = float(calculate_perimeter(mask))

        # 圆度
        circularity = (
            (4 * np.pi * cell_area) / ((cell_perimeter ** 2) + 1e-6)
            if cell_area > 0 else 0.0
        )

        # bounding box 特征
        x, y, w, h = cv2.boundingRect(mask)
        aspect_ratio   = w / h if h > 0 else 0.0
        rectangularity = cell_area / (w * h + 1e-6)

        # convex hull 特征
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 0.0
        eccentricity = 0.0
        orientation_angle = 0.0
        if len(contours) > 0:
            hull = cv2.convexHull(contours[0])
            convex_area = cv2.contourArea(hull)
            solidity = cell_area / (convex_area + 1e-6)

            # 椭圆拟合特征
            if len(contours[0]) >= 5:
                (center, axes, angle) = cv2.fitEllipse(contours[0])
                major_axis = max(axes)
                minor_axis = min(axes)
                eccentricity = np.sqrt(1 - (minor_axis / (major_axis + 1e-6)) ** 2)
                orientation_angle = angle  # 椭圆主轴角度 (0~180°)

        # 边界复杂度
        complexity = (cell_perimeter ** 2) / (cell_area + 1e-6)

        cell_info.append([
            int(idx + 1),            # index
            int(cell_area),          # area
            int(round(cell_perimeter)),  # perimeter
            float(circularity),      # circularity
            float(aspect_ratio),     # 长宽比
            float(rectangularity),   # 矩形度
            float(solidity),         # 凸实度
            float(eccentricity),     # 偏心率
            float(orientation_angle),# 主轴角度
            float(complexity),       # 边界复杂度
        ])

    return centroids, cell_info
# ===== build_and_prune_graph: PPP 公式 + 原始流程 =====
def build_and_prune_graph(
    centroids, cell_info, H, W,
    p0: float = 0.85,
    max_out_degree: int = 4,
    prune_factor: float = 2,
    mutual_confirmation: bool = True,
    logger=None, debug: bool=False
):
    """
    返回 (edges_pruned, Lmax)
    - Lmax = sqrt(-ln(1-p0) / (pi * lambda)),  lambda = N / (H*W)
    - 构图用 r*<=Lmax；剪枝再用 dist<=prune_factor*Lmax
    """
    import math, numpy as np

    N = len(centroids)
    A = float(H) * float(W)
    lam = (N / A) if A > 0 else 0.0
    Lmax = math.sqrt(-math.log(1.0 - p0) / (math.pi * lam)) if lam > 0 else 0.0

    if logger:
        diag = math.hypot(H, W)
        logger.info(f"[DBG] N={N} HxW={H}x{W}  lam={lam:.3e}  Lmax={Lmax:.1f}  diag={diag:.1f}")

    # 1) 初始边（原始版逻辑）
    nodes, edges = build_graph_dcse(centroids, cell_info, Lmax)
    if logger:
        logger.info(f"  - 剪枝前={len(edges)}")

    # 2) 剪枝
    max_dist_px = (prune_factor * Lmax) if Lmax > 0 else None
    edges_pruned = prune_edges(
        edges,
        max_out_degree=max_out_degree,
        max_dist_px=max_dist_px,
        mutual_confirmation=mutual_confirmation,
    )

    if logger:
        if edges:
            dists = [e["dist"] for e in edges]
            over  = sum(d > (max_dist_px or 1e9) for d in dists)
            logger.info(f"[DBG] pre edges: {len(edges)} | dist min/mean/max="
                        f"{min(dists):.1f}/{np.mean(dists):.1f}/{max(dists):.1f} | >cut={over}")
        logger.info(f"  - 构图完成: 剪枝后={len(edges_pruned)} (Lmax={Lmax:.2f})")

    return edges_pruned, Lmax
