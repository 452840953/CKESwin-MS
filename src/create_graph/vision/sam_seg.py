from __future__ import annotations
import cv2
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segment_anything import SamPredictor

def load_sam(checkpoint_path: str, device: str):
    from segment_anything import sam_model_registry, SamPredictor
    model = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    model.to(device=device)
    predictor = SamPredictor(model)
    return predictor

def segment_with_boxes(predictor: SamPredictor, image_path: str, boxes_xyxy):
    """Return list of binary masks aligned to original image size."""
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    H, W = img_rgb.shape[:2]
    masks_out = []
    for (xmin, ymin, xmax, ymax) in boxes_xyxy:
        box_np = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np[None, :],
            multimask_output=False,
        )
        mask = (masks[0] * 255).astype(np.uint8)
        mask = cv2.resize(mask, (W, H))
        masks_out.append(mask)
    return masks_out
