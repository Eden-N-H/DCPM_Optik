import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

"""
Defect Inspector — YOLO + SAM3 segmentation backend
Flask server that accepts an image + YOLO .pt model, runs detection,
then uses SAM3 to produce pixel-precise defect masks.
"""

import base64
import tempfile
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from ultralytics import YOLO

import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "defect_inspector"
UPLOAD_DIR.mkdir(exist_ok=True)

# YOLO uses MPS/CUDA if available (fast)
# SAM3 is hardcoded to CPU — MPS crashes on mixed dtype matmul
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SAM3_DEVICE = "cpu"

_sam3_processor: Sam3Processor | None = None


def get_sam3_processor() -> Sam3Processor:
    """Lazy-load SAM3 on CPU, forcing all weights to float32."""
    global _sam3_processor
    if _sam3_processor is None:
        print("[SAM3] Loading model from HuggingFace (this may take a moment)…")
        model = build_sam3_image_model(
            load_from_HF=True,
            device=SAM3_DEVICE,
        )
        # Cast ALL tensors to float32 to avoid bfloat16/float32 dtype mismatch
        model = model.float()
        model.to(device=SAM3_DEVICE, dtype=torch.float32)
        model.eval()
        _sam3_processor = Sam3Processor(model, device=SAM3_DEVICE)
        print(f"[SAM3] Ready on {SAM3_DEVICE}.")
    return _sam3_processor


def encode_image_b64(arr: np.ndarray) -> str:
    """Encode a BGR numpy array as a base64 PNG data-URI."""
    _, buf = cv2.imencode(".png", arr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def mask_to_contour_points(mask: np.ndarray) -> list[list[int]]:
    """Return the largest contour of a binary mask as [[x,y], …]."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).tolist()


def draw_results(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Overlay YOLO boxes + SAM3 masks on a copy of the image."""
    out = image_bgr.copy()
    colours = [
        (57, 255, 20),
        (255, 80, 80),
        (80, 180, 255),
        (255, 210, 40),
        (200, 80, 255),
    ]

    for i, det in enumerate(detections):
        colour = colours[i % len(colours)]
        x1, y1, x2, y2 = det["box"]

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        if det.get("mask") is not None:
            mask = np.array(det["mask"], dtype=np.uint8)
            overlay = out.copy()
            overlay[mask == 1] = [c // 3 for c in colour]
            cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)

            pts = np.array(det.get("contour", []), dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, colour, 2)

        label = f"{det['class']} {det['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(
            out, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
        )

    return out


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE, "sam3_device": SAM3_DEVICE})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "image" not in request.files or "model" not in request.files:
            return jsonify({"error": "Both 'image' and 'model' files are required."}), 400

        conf_threshold = float(request.form.get("conf", 0.4))

        run_id = uuid.uuid4().hex[:8]

        image_file = request.files["image"]
        image_path = UPLOAD_DIR / f"{run_id}_image{Path(image_file.filename).suffix}"
        image_file.save(image_path)

        model_file = request.files["model"]
        model_path = UPLOAD_DIR / f"{run_id}_model.pt"
        model_file.save(model_path)

        # ── Load image ────────────────────────────────────────────────────────
        pil_image = Image.open(image_path).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]

        # ── Run YOLO (MPS/CUDA/CPU) ───────────────────────────────────────────
        print(f"[YOLO] Running detection on {image_path.name}…")
        yolo_model = YOLO(str(model_path))
        yolo_results = yolo_model(pil_image, conf=conf_threshold, verbose=False)[0]

        boxes_xyxy  = yolo_results.boxes.xyxy.cpu().numpy().astype(int)
        confidences = yolo_results.boxes.conf.cpu().numpy()
        class_ids   = yolo_results.boxes.cls.cpu().numpy().astype(int)
        class_names = yolo_model.names

        if len(boxes_xyxy) == 0:
            return jsonify({
                "detections":     [],
                "result_image":   encode_image_b64(image_bgr),
                "original_image": encode_image_b64(image_bgr),
                "sam_prompt":     "",
                "message":        "No defects detected at the chosen confidence threshold.",
            })

        print(f"[YOLO] {len(boxes_xyxy)} detections found.")

        # ── Build SAM3 prompt ─────────────────────────────────────────────────
        user_prompt = request.form.get("prompt", "").strip()
        detected_classes = list({class_names[cid] for cid in class_ids})
        sam_prompt = user_prompt or ", ".join(detected_classes)
        print(f"[SAM3] Prompt: '{sam_prompt}'")

        # ── Run SAM3 (CPU, float32, no autocast) ─────────────────────────────
        processor = get_sam3_processor()

        with torch.no_grad():
            state  = processor.set_image(pil_image)
            output = processor.set_text_prompt(state=state, prompt=sam_prompt)

        sam_masks  = output["masks"].cpu().numpy()
        sam_boxes  = output["boxes"].cpu().numpy()
        sam_scores = output["scores"].cpu().numpy()

        # ── Match SAM3 masks to YOLO boxes (IoB) ─────────────────────────────
        detections = []
        for i, (box, conf, cid) in enumerate(zip(boxes_xyxy, confidences, class_ids)):
            x1, y1, x2, y2 = box
            yolo_area = max((x2 - x1) * (y2 - y1), 1)

            best_mask  = None
            best_score = -1.0
            best_iob   = 0.0

            for smask, sbox, sscore in zip(sam_masks, sam_boxes, sam_scores):
                ix1 = max(x1, int(sbox[0]))
                iy1 = max(y1, int(sbox[1]))
                ix2 = min(x2, int(sbox[2]))
                iy2 = min(y2, int(sbox[3]))
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                iob   = inter / yolo_area

                if iob > best_iob or (iob == best_iob and sscore > best_score):
                    best_iob   = iob
                    best_score = float(sscore)
                    best_mask  = smask[:h, :w].astype(np.uint8)

            mask_list   = best_mask.tolist() if best_mask is not None else None
            contour_pts = mask_to_contour_points(best_mask) if best_mask is not None else []

            defect_area_px = int(best_mask.sum()) if best_mask is not None else 0
            area_pct       = round(defect_area_px / (h * w) * 100, 3)

            detections.append({
                "id":         i,
                "class":      class_names[cid],
                "confidence": round(float(conf), 4),
                "box":        [int(x1), int(y1), int(x2), int(y2)],
                "mask":       mask_list,
                "contour":    contour_pts,
                "area_px":    defect_area_px,
                "area_pct":   area_pct,
                "sam_iob":    round(best_iob, 4),
                "sam_score":  round(best_score, 4),
            })

        # ── Render output ─────────────────────────────────────────────────────
        result_bgr   = draw_results(image_bgr, detections)
        result_b64   = encode_image_b64(result_bgr)
        original_b64 = encode_image_b64(image_bgr)

        detections_meta = [
            {k: v for k, v in d.items() if k != "mask"} for d in detections
        ]

        image_path.unlink(missing_ok=True)
        model_path.unlink(missing_ok=True)

        return jsonify({
            "detections":     detections_meta,
            "result_image":   result_b64,
            "original_image": original_b64,
            "sam_prompt":     sam_prompt,
            "image_size":     {"width": w, "height": h},
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    print(f"[Defect Inspector] Starting on http://localhost:5001  (YOLO={DEVICE}, SAM3={SAM3_DEVICE})")
    app.run(host="0.0.0.0", port=5001, debug=False)