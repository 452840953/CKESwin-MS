    # -*- coding: utf-8 -*-
"""
WoodCellsGraphDataset (cleaned & stabilized)
- Fix: consistent node feature dimension across all graphs (prevents PyG collate error)
- Reorder: build edges first, then compute node-angle variance feature
- Safety: robust edge normalization; supports empty graphs while keeping feature dims consistent
- Geometry: use image H,W for normalization (instead of pos max)
- Edge attr: include cos/sin direction; documented feature layout
- Logging tightened

PyG version: works with >=2.3; uses InMemoryDataset.save/load API.
"""

import os
import json
import cv2
import torch
import traceback as _tb
from typing import Dict, List, Tuple
from torch_geometric.data import InMemoryDataset, Data

# ==== your modules ====
from create_graph.config import YOLO_MODEL_PATH, SAM_CHECKPOINT, DEVICE, SAVEROOT, IMAGEROOT, check_paths
from create_graph.vision.detect import run_yolo
from create_graph.vision.sam_seg import segment_with_boxes
from util.logger_utils import setup_logger
from util.graph_create import _to_uint8_mask,_normalize_edges,list_images_with_labels,_build_edge_attr,build_and_prune_graph
from util.graph_create import _build_node_base_features,_compute_node_topo_feats,_build_edge_index,extract_features_from_masks

logger = setup_logger(log_dir="logs", log_prefix="graph_build")

# 预训练模型加载（分割模型+检测模型）
SAM_PREDICTOR = None
YOLO_MODEL = None

def _get_anatomy_models():
    global SAM_PREDICTOR, YOLO_MODEL
    if SAM_PREDICTOR is None:
        from create_graph.vision.sam_seg import load_sam
        SAM_PREDICTOR = load_sam(SAM_CHECKPOINT, DEVICE)
    if YOLO_MODEL is None:
        from create_graph.vision.detect import load_yolo_model
        YOLO_MODEL = load_yolo_model(YOLO_MODEL_PATH)
    return SAM_PREDICTOR, YOLO_MODEL

# ====== 节点特征定义 ======
NODE_BASE_DIM  = 9  # 基础几何特征:
# [area, perimeter, circularity,
#  aspect_ratio, rectangularity,
#  solidity, eccentricity,
#  orientation_angle, complexity]

NODE_TOPO_DIM = 8  # 拓扑 & 位置特征:
# [degree, mean_neighbor_dist,
#  mean_angle_diff, angle_std,
#  norm_x, norm_y,
#  abs_x, abs_y]

NODE_FEAT_DIM  = NODE_BASE_DIM + NODE_TOPO_DIM  # = 17 (固定)

# ====== 边特征定义 ======
EDGE_FEAT_DIM  = 9  # 边特征:
# [dist, norm_dist_img, cos, sin,
#  norm_dist_diam, diff_area, diff_circ,
#  diff_ratio, edge_angle_mean_dev]
import math
import numpy as np

# ====== 图级特征维度 ======
GRAPH_FEAT_DIM = 11

# ====== 图级特征维度 ======
# ====== 图级特征维度 ======
GRAPH_FEAT_NAMES = [
    "count_density",       # N / (H*W)
    "area_fraction",       # sum(area) / (H*W)   ← 新增
    "size_cv",
    "anisotropy",
    "ring_porosity_corr",
    "edge_len_mean",
    "edge_len_cv",
    "edge_align_cos2",
    "nbr_size_contrast",
    "Lmax_norm",
    "N"
]

# 更新节点总维度（广播拼到 x 上）
NODE_FEAT_DIM  = NODE_BASE_DIM + NODE_TOPO_DIM + GRAPH_FEAT_DIM

# —— 在你的 _save_graph_preview 里这样调用 ——
def save_graph_preview_from_index(
    save_path: str, H: int, W: int,
    pos: torch.Tensor, edge_index: torch.Tensor,
    node_radius: int = 2, line_thickness: int = 2,
    draw_indices: bool = True, index_base: int = 1,
    font_scale: float = 0.5, font_thickness: int = 1,
    label_offset: Tuple[int, int] = (4, -4),
    undirected_draw: bool = True  # 只画一次无向边
):
    """
    用 edge_index (0-based) 可视化图结构。和 GNN 完全一致。
    - pos.shape = [N, 2]
    - edge_index.shape = [2, E]，0-based 索引
    """
    import numpy as np, cv2

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # --- 画边（用 edge_index，0-based） ---
    if (edge_index is not None and edge_index.numel() > 0 and
        pos is not None and pos.numel() > 0):
        src = edge_index[0].detach().cpu().numpy().astype(np.int64)
        dst = edge_index[1].detach().cpu().numpy().astype(np.int64)
        P   = pos.detach().cpu().numpy().astype(np.int32)
        n   = P.shape[0]

        drawn = set()
        for i, j in zip(src, dst):
            if i == j:
                continue
            # 跳过越界（若上游有脏数据）
            if not (0 <= i < n and 0 <= j < n):
                continue
            key = (min(i, j), max(i, j)) if undirected_draw else (i, j)
            if key in drawn:
                continue
            drawn.add(key)

            x1, y1 = P[i]; x2, y2 = P[j]
            x1 = int(np.clip(x1, 0, W-1)); y1 = int(np.clip(y1, 0, H-1))
            x2 = int(np.clip(x2, 0, W-1)); y2 = int(np.clip(y2, 0, H-1))
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 200, 0),
                     thickness=line_thickness, lineType=cv2.LINE_AA)

    # --- 画点 + 序号（序号仅用于显示，index_base=1 则显示 1..N） ---
    if pos is not None and pos.numel() > 0:
        P = pos.detach().cpu().numpy().astype(np.int32)
        for idx, (x, y) in enumerate(P):
            x = int(np.clip(x, 0, W-1)); y = int(np.clip(y, 0, H-1))
            cv2.circle(canvas, (x, y), node_radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            if draw_indices:
                lbl = str(idx + index_base)
                tx = int(np.clip(x + label_offset[0], 0, W-1))
                ty = int(np.clip(y + label_offset[1], 0, H-1))
                cv2.putText(canvas, lbl, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                            thickness=font_thickness, lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)

def save_graph_preview_from_edges(
    save_path, H, W, pos, edges,
    node_radius=2, line_thickness=1,
    draw_indices=True, index_base=1,
    font_scale=0.5, font_thickness=1, label_offset=(4,-4)
):
    import numpy as np, cv2, torch
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # --- 画边：按 edges_pruned（1-based），无向去重 ---
    if edges and pos is not None and pos.numel() > 0:
        P = pos.detach().cpu().numpy().astype(np.int32)
        drawn = set()
        for e in edges:
            i = e["src"] - 1
            j = e["dst"] - 1
            if i == j:
                continue
            if not (0 <= i < len(P) and 0 <= j < len(P)):
                continue
            key = (min(i, j), max(i, j))
            if key in drawn:
                continue
            drawn.add(key)
            x1, y1 = P[i]; x2, y2 = P[j]
            x1 = int(np.clip(x1, 0, W-1)); y1 = int(np.clip(y1, 0, H-1))
            x2 = int(np.clip(x2, 0, W-1)); y2 = int(np.clip(y2, 0, H-1))
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 200, 0),
                     thickness=line_thickness, lineType=cv2.LINE_AA)

    # --- 画点 & 序号 ---
    if pos is not None and pos.numel() > 0:
        P = pos.detach().cpu().numpy().astype(np.int32)
        for idx, (x, y) in enumerate(P):
            x = int(np.clip(x, 0, W-1)); y = int(np.clip(y, 0, H-1))
            cv2.circle(canvas, (x, y), node_radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            if draw_indices:
                lbl = str(idx + index_base)
                tx = int(np.clip(x + label_offset[0], 0, W-1))
                ty = int(np.clip(y + label_offset[1], 0, H-1))
                cv2.putText(canvas, lbl, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0),
                            thickness=font_thickness, lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)

def _graph_feat_dict(gf_vec: torch.Tensor) -> Dict[str, float]:
    """把 (GRAPH_FEAT_DIM,) 的张量转成 {name: value} 方便打印/记录"""
    arr = gf_vec.detach().cpu().numpy().tolist()
    return {k: float(arr[i]) for i, k in enumerate(GRAPH_FEAT_NAMES)}

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return 0.0
    sx, sy = x.std(ddof=0), y.std(ddof=0)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    xz, yz = (x - x.mean()) / sx, (y - y.mean()) / sy
    return float((xz * yz).mean())

def _compute_graph_level_features(
    pos: torch.Tensor,
    x_base: torch.Tensor,
    edge_index: torch.Tensor,
    hw: Tuple[int, int],
    Lmax: float = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      - gf_node: (N, GRAPH_FEAT_DIM)  图级特征广播到节点
      - gf_vec : (GRAPH_FEAT_DIM,)    单个图的图级向量（方便以后模型使用）
    仅依赖: 节点质心 pos、基础几何 x_base[:,0]=area、边 edge_index、图像尺寸 hw
    """
    H, W = hw
    N = int(pos.size(0))
    diag = float(math.hypot(H, W))
    if N == 0:
        gf_vec = torch.zeros((GRAPH_FEAT_DIM,), dtype=torch.float32)
        return torch.empty((0, GRAPH_FEAT_DIM), dtype=torch.float32), gf_vec

    # 1) 基于面积的等效直径
    area = x_base[:, 0].detach().cpu().numpy().astype(np.float64)  # 假设第0列是 area
    eq_diam = 2.0 * np.sqrt(np.clip(area, 0.0, None) / math.pi)    # 直径

    # 2) 节点密度（按像素面积归一）
    count_density = N / float(H * W + 1e-12)
    area_fraction = float(area.sum() / float(H * W + 1e-12))  # ← 孔隙率
    
    # 3) 位置分布各向异性（2x2 协方差的特征值比）
    P = pos.detach().cpu().numpy().astype(np.float64)    # (N,2)
    Pc = P - P.mean(axis=0, keepdims=True)
    if N >= 2:
        C = np.cov(Pc.T)                                 # 2x2
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 0.0, None)
        lam_max, lam_min = float(w.max()), float(w.min() + 1e-12)
        anisotropy = (lam_max - lam_min) / (lam_max + lam_min + 1e-12)  # ∈[0,1)
        # 主轴方向（最大特征值对应的特征向量）
        u = V[:, int(np.argmax(w))].astype(np.float64)
        u = u / (np.linalg.norm(u) + 1e-12)
    else:
        anisotropy = 0.0
        u = np.array([1.0, 0.0], dtype=np.float64)

    # 4) “环孔倾向”相关：沿主轴坐标与孔径的皮尔逊相关
    t = Pc @ u  # 投影到主轴
    ring_porosity_corr = _safe_corr(t, eq_diam)

    # 5) 孔径变异系数（全图）
    size_cv = float(eq_diam.std(ddof=0) / (eq_diam.mean() + 1e-12))

    # 6~7) 边长度统计（归一化到对角线）
    if edge_index.numel() > 0:
        src = edge_index[0].detach().cpu().numpy()
        dst = edge_index[1].detach().cpu().numpy()
        elen = np.linalg.norm(P[dst] - P[src], axis=1) / (diag + 1e-12)
        edge_len_mean = float(elen.mean())
        edge_len_cv   = float(elen.std(ddof=0) / (elen.mean() + 1e-12))
    else:
        edge_len_mean = 0.0
        edge_len_cv   = 0.0

    # 8) 全图边方向对齐度（相对主轴），<cos(2θ)>，旋转不敏感
    if edge_index.numel() > 0:
        v = P[dst] - P[src]
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        cos_theta = (v @ u.reshape(2,1)).ravel()
        edge_align_cos2 = float(np.mean(2.0 * cos_theta**2 - 1.0))  # ∈[-1,1]
    else:
        edge_align_cos2 = 0.0

    # 9) 邻接尺寸对比（边两端面积的归一化差）
    if edge_index.numel() > 0:
        ai, aj = area[src], area[dst]
        size_contrast = np.abs(ai - aj) / (ai + aj + 1e-12)
        nbr_size_contrast = float(size_contrast.mean())
    else:
        nbr_size_contrast = 0.0

    # 10) Lmax 归一化（构图阶段返回的尺度标尺，若无则置 0）
    Lmax_norm = float((0.0 if Lmax is None else Lmax) / (diag + 1e-12))

    # 组装向量
    gf = np.array([
        count_density,
        area_fraction,  # ← 新增到向量
        size_cv,
        anisotropy,
        ring_porosity_corr,
        edge_len_mean,
        edge_len_cv,
        edge_align_cos2,
        nbr_size_contrast,
        Lmax_norm,
        float(N)
    ], dtype=np.float32)

    gf_vec = torch.from_numpy(gf)                 # (GRAPH_FEAT_DIM,)
    gf_node = gf_vec.unsqueeze(0).repeat(N, 1)    # 广播到 (N, D)
    return gf_node, gf_vec

# ---------- main build ----------
def build_pyg_data_from_image(
    image_path: str,
    skip_cls: int = 1,
    undirected: bool = True,
    add_edge_attr: bool = True,
    verbose: bool = True,
    predictor=None,
    yolo_model=None,
) -> Data:
    if verbose:
        logger.info(f"\n[STEP] 构图: {image_path}")

    check_paths()
    if predictor is None or yolo_model is None:
        default_predictor, default_detector = _get_anatomy_models()
        predictor = default_predictor if predictor is None else predictor
        yolo_model = default_detector if yolo_model is None else yolo_model

    # image size
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"cv2.imread 失败: {image_path}")
    H, W = img.shape[:2]

    # 1) YOLO
    save_dir1, dets = run_yolo(yolo_model, image_path)
    if verbose:
        logger.info(f"  - YOLO检测完毕: dets={len(dets)}，输出目录={save_dir1}")

    boxes = []
    for xmin, ymin, xmax, ymax, conf, cls in dets:
        if int(cls) == skip_cls:
            continue
        boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
    if verbose:
        logger.info(f"  - 筛后保留框数: {len(boxes)}")

    # 2) SAM
    masks_raw = segment_with_boxes(predictor, image_path, boxes)
    if verbose:
        logger.info(f"  - SAM分割得到 masks: {len(masks_raw)}")

    # 3) features per cell
    # centroids: List[Tuple[int,int]] = []
    # cell_info: List[List[float]] = []
    centroids, cell_info = extract_features_from_masks(masks_raw)
    # 转tensor好计算
    num_nodes = len(centroids)
    pos = torch.tensor(centroids, dtype=torch.float32) if num_nodes > 0 else torch.empty((0, 2), dtype=torch.float32)
    x_base = _build_node_base_features(cell_info)   # (N,8)

    if verbose:
        logger.info(f"  - 特征提取: 节点数={num_nodes}")

    # 4) graph & prune
    edges_pruned, Lmax = build_and_prune_graph(
        centroids,
        cell_info,
        H,
        W,
        p0=0.95,
        max_out_degree=4,
        prune_factor=2,
        mutual_confirmation=True,
        logger=logger,
        debug=True
    )

    if verbose:
        logger.info(f"  - 构图完成: 剪枝后={len(edges_pruned)} (Lmax={Lmax:.2f})")
    # 5) edge index after pruning
    edge_index = _build_edge_index(edges_pruned, num_nodes=num_nodes, undirected=undirected, verbose=verbose, logger=logger)

    # 6) node extra feature (ALWAYS add to keep (N,9), even when N=0)
    node_topo_var = _compute_node_topo_feats(pos, edge_index, hw=(H, W))  # ✅
    # 6.5) graph-level features（广播 + 保留1D向量）
    gf_node, gf_vec = _compute_graph_level_features(
        pos=pos, x_base=x_base, edge_index=edge_index, hw=(H, W), Lmax=Lmax
    )
    
    # ★ 打印当前图的图级特征
    if verbose:
        logger.info("  - Graph-level features:\n" + json.dumps(_graph_feat_dict(gf_vec), ensure_ascii=False, indent=2))
    
    if x_base.numel() > 0:
        x = torch.cat([x_base, node_topo_var, gf_node], dim=1)
    else:
        x = torch.empty((0, NODE_FEAT_DIM), dtype=torch.float32)
    # x = torch.cat([x_base, node_angle_var], dim=1) if x_base.numel() > 0 else torch.empty((0, NODE_FEAT_DIM), dtype=torch.float32)

    # 7) edge attributes (fixed dim)
    edge_attr = _build_edge_attr(pos, x, edge_index, hw=(H, W)) if add_edge_attr else None

    if verbose:
        logger.info(f"  -> 成功: 节点={num_nodes}, 边数={edge_index.size(1)} | x.shape={tuple(x.shape)} edge_attr.shape={(tuple(edge_attr.shape) if edge_attr is not None else None)}")

    data = Data(
        x=x,
        edge_index=edge_index,
        pos=pos,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
        global_feat=gf_vec,
        global_feat_names=GRAPH_FEAT_NAMES
    )
    return data,Lmax,edges_pruned

class WoodCellsGraphDataset(InMemoryDataset):
    """Read images from data_root (each subfolder is a class) and build PyG graphs.
    Saves to ROOT/processed/data.pt
    """
    def __init__(
        self,
        root: str,
        data_root: str = "./data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        undirected: bool = True,
        add_edge_attr: bool = True,
        skip_cls: int = 1,
        verbose: bool = True,
    ):
        self.data_root     = data_root
        self.undirected    = undirected
        self.add_edge_attr = add_edge_attr
        self.skip_cls      = skip_cls
        self.verbose       = verbose

        self.image_paths, self.labels, self.class_to_idx = list_images_with_labels(self.data_root)

        if self.verbose:
            logger.info(f"[INFO] 类别映射(class_to_idx): {self.class_to_idx}")
            logger.info(f"[INFO] 共计图片: {len(self.image_paths)}")

        super().__init__(root, transform, pre_transform, pre_filter)

        if os.path.exists(self.processed_paths[0]):
            logger.info("数据被发现，将直接加载")

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        classes_json = os.path.join(self.processed_dir, "class_to_idx.json")
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(classes_json, "w", encoding="utf-8") as f:
            json.dump(self.class_to_idx, f, ensure_ascii=False, indent=2)

        data_list = []
        total = len(self.image_paths)
        for i, (img_path, y_idx) in enumerate(zip(self.image_paths, self.labels), start=1):
            if self.verbose:
                logger.info(f"\n[PROGRESS] ({i}/{total}) 处理: {img_path} | 类别={y_idx}")
            try:
                d,Lmax,edges_pruned = build_pyg_data_from_image(
                    img_path,
                    skip_cls=self.skip_cls,
                    undirected=self.undirected,
                    add_edge_attr=self.add_edge_attr,
                    verbose=self.verbose,
                    predictor=SAM_PREDICTOR,
                    yolo_model=YOLO_MODEL,
                )
                d.y = torch.tensor([y_idx], dtype=torch.long)
                d.img_path = img_path
                
                # === 生成并保存该图的网络示意图 ===
                try:
                    # 用原图尺寸，保持一致
                    im0 = cv2.imread(img_path)
                    if im0 is not None:
                        H, W = im0.shape[:2]
                    else:
                        # 回退：用点坐标范围估一个安全画布
                        if d.pos is not None and d.pos.numel() > 0:
                            W = int(max(1, d.pos[:, 0].max().item() + 5))
                            H = int(max(1, d.pos[:, 1].max().item() + 5))
                        else:
                            H, W = 512, 512  # 全空图时的回退尺寸
                    
                    # 在 root 下建立 Graph_img，并复用 data_root 的相对路径，避免同名冲突
                    rel_path = os.path.relpath(img_path, start=self.data_root)  # e.g. classA/xxx.jpg
                    stem, _ = os.path.splitext(rel_path)
                    out_png = os.path.join(self.root, "Graph_img", stem + "_graph.png")
                    
                    save_graph_preview_from_index(
                        out_png, H, W, d.pos, d.edge_index,  # ← 用 edge_index
                        node_radius=2, line_thickness=2, draw_indices=True,
                        index_base=1, undirected_draw=True
                    )
                    if self.verbose:
                        logger.info(f"  - Graph preview saved: {out_png}")
                except Exception as _e:
                    logger.warning(f"  - Graph preview failed: {repr(_e)}")
                
                # 可选：跳过空图（如果你不希望把0节点样本放进训练）
                # if d.num_nodes == 0:
                #     if self.verbose:
                #         logger.info("  -> 空图，跳过")
                #     continue

                # 断言：保证维度一致
                assert d.x.size(1) == NODE_FEAT_DIM, f"x dim mismatch: {d.x.size()}"
                if d.edge_attr is not None:
                    assert d.edge_attr.size(1) == EDGE_FEAT_DIM, f"edge_attr dim mismatch: {d.edge_attr.size()}"

                data_list.append(d)
                if self.verbose:
                    logger.info(f"  -> 成功: 节点={d.num_nodes}, 边数={d.edge_index.size(1)}")
            except Exception:
                logger.error("  -> 失败，已跳过.")
                logger.error(_tb.format_exc())
                continue

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])
        if self.verbose:
            logger.info(f"\n[OK] 已保存到: {self.processed_paths[0]}")
            logger.info(f"[OK] 类别映射写入: {classes_json}")

# ====== example ======
if __name__ == "__main__":
    
    logger.info("\n=== 程序开始执行 ===")
    ROOT = SAVEROOT
    dataset = WoodCellsGraphDataset(root=SAVEROOT, data_root=IMAGEROOT, verbose=True)

    logger.info("\n=== 数据集信息 ===")
    logger.info(dataset)
    logger.info(f"图数量: {len(dataset)}")
    if len(dataset) > 0:
        logger.info(f"样例图 Data： {dataset[0]}")
        logger.info(f"样例图 y： {dataset[0].y}  (类别索引)")
