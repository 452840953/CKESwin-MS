import os
from pathlib import Path

# You can override via environment variables:
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "external_weights/vessel_detector.pt")
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "external_weights/sam_vit_h.pth")
DEVICE = os.getenv("DEVICE", "cuda")
SAVEROOT = os.getenv("SAVEROOT", "inputs/graphs")
IMAGEROOT = os.getenv("IMAGEROOT", "inputs/images")

# Validate (warn only; we won't raise to keep UI usable)
def check_paths():
    msgs = []
    if not Path(YOLO_MODEL_PATH).exists():
        msgs.append(f"[WARN] YOLO model not found at: {YOLO_MODEL_PATH}")
    if not Path(SAM_CHECKPOINT).exists():
        msgs.append(f"[WARN] SAM checkpoint not found at: {SAM_CHECKPOINT}")
    if msgs:
        print("\\n".join(msgs))
