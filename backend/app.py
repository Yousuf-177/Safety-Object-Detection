# app.py (Render-safe YOLO backend)

import os
import io
import uuid
import logging
from typing import List, Dict, Any
from collections import Counter
from threading import Lock

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

# ==========================
# CONFIG
# ==========================
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
USE_CUDA = os.environ.get("USE_CUDA", "0") == "1"
DEVICE = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.25))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", 0.45))

CLASS_NAMES = [
    "OxygenTank",
    "NitrogenTank",
    "FirstAidBox",
    "FireAlarm",
    "SafetySwitchPanel",
    "EmergencyPhone",
    "FireExtinguisher",
]

MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", 5))
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

OUTPUT_DIR = "/tmp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# APP INIT
# ==========================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES

CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("safety-backend")

# ==========================
# MODEL LOADING
# ==========================
def load_model(path: str):
    log.info("Loading YOLO model from %s", path)
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        log.info("Loaded model using ultralytics")
        return model
    except Exception as e:
        log.warning("Ultralytics load failed: %s", e)

    try:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=path, force_reload=False)
        model.to(DEVICE)
        log.info("Loaded model using torch.hub fallback")
        return model
    except Exception as e:
        log.exception("Model load failed")
        raise RuntimeError("Model load failed") from e


MODEL = load_model(MODEL_PATH)
try:
    MODEL.eval()
except Exception:
    try:
        MODEL.model.eval()
    except Exception:
        pass

log.info("Model ready on device: %s", DEVICE)

# ==========================
# FONT LOAD (ONCE)
# ==========================
BASE_DIR = os.path.dirname(__file__)
FONT_PATH = os.path.join(BASE_DIR, "fonts", "DejaVuSans-Bold.ttf")

try:
    FONT = ImageFont.truetype(FONT_PATH, size=28)
    log.info("Font loaded: %s", FONT_PATH)
except Exception as e:
    log.warning("Font load failed, using default: %s", e)
    FONT = ImageFont.load_default()

# ==========================
# LOCK (PREVENT CONCURRENT GPU/CPU SPIKES)
# ==========================
model_lock = Lock()

# ==========================
# DRAWING UTILS
# ==========================
def draw_detections(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = FONT

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # Pillow 10+ safe text size
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = font.getsize(label)

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=4)

        # Label background
        draw.rectangle(
            [x1, y1 - text_h - 8, x1 + text_w + 8, y1],
            fill="black"
        )

        # Text
        draw.text((x1 + 4, y1 - text_h - 6), label, fill="white", font=font)

    return image

# ==========================
# INFERENCE
# ==========================
def run_inference_pil(image: Image.Image, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    global MODEL
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    img_np = np.array(image)

    with model_lock:
        try:
            # Render-safe ultralytics inference
            if hasattr(MODEL, "predict"):
                results = MODEL.predict(
                    source=img_np,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    imgsz=640,
                    device="cpu",
                    verbose=False
                )
            else:
                results = MODEL(img_np, conf=conf_thresh, iou=iou_thresh)

            res0 = results[0] if isinstance(results, (list, tuple)) else results
            boxes = getattr(res0, "boxes", None)

            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                dets = []
                for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
                    if conf < conf_thresh:
                        continue

                    dets.append({
                        "class_id": int(cid),
                        "class_name": CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES) else str(int(cid)),
                        "confidence": float(conf),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
                return dets
        except Exception as e:
            log.exception("Ultralytics inference failed")

        try:
            # Torch hub fallback
            results = MODEL(img_np, size=640)
            xyxy = results.xyxy[0].cpu().numpy()

            dets = []
            for row in xyxy:
                x1, y1, x2, y2, conf, cid = row
                if conf < conf_thresh:
                    continue

                dets.append({
                    "class_id": int(cid),
                    "class_name": CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES) else str(int(cid)),
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })
            return dets
        except Exception:
            log.exception("Torch hub inference failed")

    raise RuntimeError("Inference failed")

# ==========================
# ROUTES
# ==========================
@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        incoming_files = request.files.getlist("images") or request.files.getlist("image")
        if not incoming_files:
            return jsonify({"ok": False, "error": "No files provided under key 'images' or 'image'."}), 400

        conf = float(request.form.get("conf_threshold", CONF_THRESHOLD))
        iou = float(request.form.get("iou_threshold", IOU_THRESHOLD))

        results_out = []

        for file_storage in incoming_files:
            if file_storage.content_length and file_storage.content_length > MAX_FILE_BYTES:
                return jsonify({"ok": False, "error": f"File too large (max {MAX_FILE_MB} MB)"}), 400

            filename = file_storage.filename or "upload"

            try:
                image = Image.open(file_storage.stream).convert("RGB")
            except Exception as e:
                log.warning("Invalid image %s: %s", filename, e)
                results_out.append({
                    "filename": filename,
                    "detections": [],
                    "summary": {"total": 0, "by_class": {}},
                    "image_url": None
                })
                continue

            detections = run_inference_pil(image, conf_thresh=conf, iou_thresh=iou)

            total = len(detections)
            by_class = dict(Counter([d["class_name"] for d in detections]))

            annotated = image.copy()
            annotated = draw_detections(annotated, detections)

            file_id = uuid.uuid4().hex
            output_name = f"{file_id}.png"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            annotated.save(output_path)

            results_out.append({
                "filename": filename,
                "detections": detections,
                "summary": {"total": total, "by_class": by_class},
                "image_url": f"/files/{output_name}"
            })

        return jsonify({"ok": True, "results": results_out})

    except Exception as e:
        log.exception("Detection error")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/files/<name>")
def serve_file(name):
    return send_from_directory(OUTPUT_DIR, name)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "model_loaded": MODEL is not None, "device": DEVICE}), 200


# ==========================
# ENTRYPOINT
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
