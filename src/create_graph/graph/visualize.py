\
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx
from skimage.measure import regionprops

def calculate_perimeter(mask):
    regions = regionprops(mask.astype(int))
    return sum(region.perimeter for region in regions)

def draw_masks_on_images(image, lumen_masks, wall_masks, save_dir):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2

    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img_with_cell  = img.copy()
    img_with_wall  = img.copy()
    img_with_lumen = img.copy()
    img_with_all   = img.copy()

    for mask in wall_masks:
        m = mask.astype(np.uint8)
        img_with_wall[m == 255] = [255, 0, 0]

    for idx, mask in enumerate(lumen_masks):
        m = mask.astype(np.uint8)
        img_with_lumen[m == 255] = [0, 0, 255]
        img_with_all[m == 255]   = [0, 0, 255]
        M = cv2.moments(m)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
            cv2.putText(img_with_all, str(idx+1), (cX - 20, cY), font, font_scale, font_color, thickness)
            cv2.putText(img_with_lumen, str(idx+1), (cX - 20, cY), font, font_scale, font_color, thickness)

    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "img_with_cell.png"),  img_with_cell)
    cv2.imwrite(os.path.join(save_dir, "img_with_wall.png"),  img_with_wall)
    cv2.imwrite(os.path.join(save_dir, "img_with_lumen.png"), img_with_lumen)
    cv2.imwrite(os.path.join(save_dir, "img_with_all.png"),   img_with_all)
    return (os.path.join(save_dir, "img_with_cell.png"),
            os.path.join(save_dir, "img_with_wall.png"),
            os.path.join(save_dir, "img_with_lumen.png"),
            os.path.join(save_dir, "img_with_all.png"))

def visualize_graph_on_image(image_path, centroids, edges, save_path,
                             node_radius=4, edge_thickness=2, edge_alpha=0.6,
                             node_color=(255, 0, 0), edge_color=(0, 255, 0),
                             show_ids=False):
    base = cv2.imread(image_path, cv2.IMREAD_COLOR)
    overlay = base.copy()

    for e in edges:
        i, j = e["src"]-1, e["dst"]-1
        x1, y1 = int(centroids[i][0]), int(centroids[i][1])
        x2, y2 = int(centroids[j][0]), int(centroids[j][1])
        cv2.line(overlay, (x1, y1), (x2, y2), edge_color, edge_thickness, lineType=cv2.LINE_AA)

    for k, (cx, cy) in enumerate(centroids, start=1):
        cx, cy = int(cx), int(cy)
        cv2.circle(overlay, (cx, cy), node_radius, node_color, -1, lineType=cv2.LINE_AA)
        if show_ids:
            cv2.putText(overlay, str(k), (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    vis = cv2.addWeighted(overlay, edge_alpha, base, 1-edge_alpha, 0)
    cv2.imwrite(save_path, vis)
    return save_path

def visualize_graph_plain(centroids, edges, save_path,
                          node_radius=4, edge_thickness=2,
                          node_color=(0, 0, 255), edge_color=(0, 255, 0),
                          text_color=(0, 0, 0), font_scale=0.5, margin=10):
    if len(centroids) == 0:
        canvas = np.full((512, 512, 3), 255, dtype=np.uint8)
        cv2.imwrite(save_path, canvas)
        return save_path

    max_x = int(max(x for x, _ in centroids)) + margin
    max_y = int(max(y for _, y in centroids)) + margin
    h = max(1, max_y); w = max(1, max_x)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    for e in edges:
        i, j = e["src"] - 1, e["dst"] - 1
        x1, y1 = map(int, centroids[i])
        x2, y2 = map(int, centroids[j])
        cv2.line(canvas, (x1, y1), (x2, y2), edge_color, edge_thickness, lineType=cv2.LINE_AA)

    for idx, (cx, cy) in enumerate(centroids, start=1):
        cx, cy = int(cx), int(cy)
        cv2.circle(canvas, (cx, cy), node_radius, node_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, str(idx), (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, canvas)
    return save_path

def visualize_graph(nodes, edges, save_path):
    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.DiGraph()
    for nid, feats in nodes.items():
        G.add_node(nid, area=feats[1])
    for e in edges:
        G.add_edge(e["src"], e["dst"], weight=e["dist"])
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue",
            font_size=8, edge_color="gray", arrowsize=12)
    nx.draw_networkx_edge_labels(G, pos,
        edge_labels={(e["src"], e["dst"]): f'{e["dist"]:.1f}' for e in edges},
        font_size=6)
    plt.axis("off"); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    return save_path
