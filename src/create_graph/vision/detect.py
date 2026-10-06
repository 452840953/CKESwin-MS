# detect.py
from create_graph.config import DEVICE

def load_yolo_model(model_path: str):
    """全局只加载一次 YOLO 模型"""
    from ultralytics import YOLO
    return YOLO(model_path)

def run_yolo(model, image_path: str):
    """
    使用已加载的 YOLO 模型推理
    返回 (save_dir, detections_info)
    """
    results = model.predict(source=image_path, save=True, save_txt=False, device=DEVICE)
    save_dir = str(results[0].save_dir)

    detections_info = []
    boxes = results[0].boxes
    for i in range(len(boxes.xyxy)):
        xmin, ymin, xmax, ymax = boxes.xyxy[i]
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        detections_info.append([xmin, ymin, xmax, ymax, conf, cls])

    return save_dir, detections_info
