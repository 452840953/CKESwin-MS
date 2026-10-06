# ===== build_graph_dcse: 原始版风格（半平面扇区 + r* 最小）=====
import math
import numpy as np

def build_graph_dcse(centroids, cell_info, Lmax, eps=1e-6):
    """
    centroids: [(x,y), ...]  图像坐标，y向下为正
    cell_info: 你的节点特征表（保持不变）
    Lmax     : r* 的阈值（建议用 PPP 推导：sqrt(-ln(1-p0)/(pi*lambda))）
    """
    N = len(centroids)
    edges = []
    nodes = {i + 1: cell_info[i] for i in range(N)}

    for i in range(N):
        xi, yi = centroids[i]
        # 每个方向存 (r_star, j, dx, dy)
        cand = {"N": [], "S": [], "E": [], "W": []}

        for j in range(N):
            if i == j:
                continue
            dx = centroids[j][0] - xi
            dy = centroids[j][1] - yi
            if dx == 0 and dy == 0:
                continue

            # —— 四个半平面（和你“原始版”一致）——
            # N: dy >= 0 且 dy >= |dx|      → r* = +dy
            if dy >= 0 and dy >= abs(dx):
                r = dy
                if r > 0: cand["N"].append((r, j, dx, dy))
            # S: dy <= 0 且 -dy >= |dx|     → r* = -dy
            if dy <= 0 and -dy >= abs(dx):
                r = -dy
                if r > 0: cand["S"].append((r, j, dx, dy))
            # E: dx >= 0 且 dx >= |dy|      → r* = +dx
            if dx >= 0 and dx >= abs(dy):
                r = dx
                if r > 0: cand["E"].append((r, j, dx, dy))
            # W: dx <= 0 且 -dx >= |dy|     → r* = -dx
            if dx <= 0 and -dx >= abs(dy):
                r = -dx
                if r > 0: cand["W"].append((r, j, dx, dy))

        # 每个方向只取 r* 最小；若 r* 打平，用欧氏距离打破平手
        for D, lst in cand.items():
            if not lst:
                continue
            r_star, j, dx, dy = min(lst, key=lambda t: (t[0], math.hypot(t[2], t[3])))
            if r_star > Lmax + eps:
                continue  # 构图阶段就过滤远连边

            dist = float(math.hypot(dx, dy))
            ang  = float(math.atan2(dy, dx))
            edges.append({
                "src": i + 1,
                "dst": j + 1,
                "r_star": float(r_star),
                "dist":  dist,
                "cos":   float(math.cos(ang)),
                "sin":   float(math.sin(ang)),
                "d_area":  float(nodes[i + 1][1]   - nodes[j + 1][1]),
                "d_circ":  float(nodes[i + 1][-2]  - nodes[j + 1][-2]),
                "d_ratio": float(nodes[i + 1][-1]  - nodes[j + 1][-1]),
            })

    return nodes, edges
