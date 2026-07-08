"""SAM2 integration for pixel-precise defect segmentation."""
import numpy as np
import cv2
import torch

_sam2_predictor = None

# 1. CHANGED THIS LINE BACK TO THE LONG CONFIG:
def load_sam2(checkpoint_path="models/sam2.1_hiera_large.pt", config="configs/sam2.1/sam2.1_hiera_l.yaml"):
    """Load SAM2 model at startup. Returns the predictor."""
    global _sam2_predictor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sam2_model = build_sam2(config_file=config, ckpt_path=checkpoint_path, device=device)
    _sam2_predictor = SAM2ImagePredictor(sam2_model)
    return _sam2_predictor

def get_predictor():
    """Get the loaded SAM2 predictor (call load_sam2 first)."""
    return _sam2_predictor

def run_sam2_on_detections(image_rgb, yolo_result, predictor=None):
    """Run SAM2 segmentation on YOLO detection bounding boxes."""
    if predictor is None:
        predictor = _sam2_predictor
    if predictor is None:
        return []
    
    if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        return []
    
    boxes = yolo_result.boxes
    model_names = yolo_result.names if hasattr(yolo_result, 'names') else {}
    
    box_array = boxes.xyxy.cpu().numpy().astype(np.float32)
    predictor.set_image(image_rgb)
    
    masks, scores, logits = predictor.predict(
        box=box_array,
        multimask_output=False,
    )
    
    results = []
    for i in range(len(boxes)):
        if masks.ndim == 4:
            mask = masks[i, 0]
        elif masks.ndim == 3:
            mask = masks[i]
        else:
            mask = masks
        
        binary_mask = (mask > 0).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cls_id = int(boxes.cls[i].cpu().numpy())
        conf = float(boxes.conf[i].cpu().numpy())
        class_name = model_names.get(cls_id, f"class_{cls_id}")
        
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue
            pts = contour.reshape(-1, 2).astype(np.float32)
            results.append((pts, cls_id, conf, class_name))
    
    return results