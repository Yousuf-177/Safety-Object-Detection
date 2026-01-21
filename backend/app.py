# app.py
import os
import io
import base64
import logging
from typing import List, Dict, Any
from collections import Counter

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# optional torch/ultralytics imports done in loader
import torch
from threading import Lock
model_lock = Lock()
# CONFIG
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" and torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.25))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", 0.45))
CLASS_NAMES = ["OxygenTank","NitrogenTank","FirstAidBox","FireAlarm","SafetySwitchPanel","EmergencyPhone","FireExtinguisher"]

# App
app = Flask(__name__)
# Optional: set max upload size (bytes). Example 10 MB
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 20 * 1024 * 1024))
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("safety-backend")

try:
    font = ImageFont.truetype(FONT_PATH, size=32)
    print("FONT LOADED:", FONT_PATH)
except Exception as e:
    print("Font load error:", e)
    font = ImageFont.load_default()

# Model loader
def load_model(path: str):
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        log.info("Loaded model with ultralytics")
        return model
    except Exception as e:
        log.info("ultralytics loader failed: %s", e)
    try:
        model = torch.hub.load('ultralytics/yolov11', 'custom', path=path, force_reload=False)
        model.to(DEVICE)
        log.info("Loaded model with torch.hub yolov11")
        return model
    except Exception as e:
        log.exception("Failed to load model: %s", e)
        raise RuntimeError("Model load failed") from e

MODEL = None

# After app = Flask(__name__)

log.info("Loading model on startup...")
MODEL = load_model(MODEL_PATH)
try:
    MODEL.eval()
except Exception:
    try:
        MODEL.model.eval()
    except Exception:
        pass
log.info("Model loaded and ready.")

import os
from PIL import ImageFont

BASE_DIR = os.path.dirname(__file__)  # folder where this file lives
FONT_PATH = os.path.join(BASE_DIR, "fonts", "DejaVuSans-Bold.ttf")



# drawing helper
def draw_detections(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    # Load font    
    font = FONT

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # ---- FIX FOR PILLOW 10+ ----
        # get text bounding box
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            # fallback if pillow < 10
            text_w, text_h = font.getsize(label)
        # --------------------------------

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="green", width=5)
        
        # Label background box
        draw.rectangle(
            [x1, y1 - text_h - 6, x1 + text_w + 6, y1],
            fill="black"
        )

        # Draw label text
        draw.text((x1 + 3, y1 - text_h - 4), label, fill="white", font=font)

    return image


# inference wrapper (support ultralytics & yolov5)
def run_inference_pil(image: Image.Image, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD):
    global MODEL
    if MODEL is None:
        raise RuntimeError("Model not loaded")
    img_np = np.array(image)

    # ultralytics path
    try:
        with model_lock:
        # ultralytics supports `predict` or direct call
            results = MODEL.predict(source=img_np,conf=conf_thresh,iou=iou_thresh,imgsz=640,device="cpu",verbose=False) if hasattr(MODEL, "predict") else MODEL(img_np, conf=conf_thresh, iou=iou_thresh)
            res0 = results[0] if isinstance(results, (list, tuple)) else results
            boxes = getattr(res0, "boxes", None)
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                dets = []
                for (x1,y1,x2,y2), conf, cid in zip(xyxy, confs, cls_ids):
                    if conf < conf_thresh: continue
                    dets.append({
                        "class_id": int(cid),
                        "class_name": CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES) else str(int(cid)),
                        "confidence": float(conf),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
                return dets
    except Exception as e:
        log.debug("Ultralytics inference failed: %s", e)

    # yolov5 torch.hub path
    try:
        results = MODEL(img_np, size=640)
        xyxy = results.xyxy[0].cpu().numpy()
        dets = []
        for row in xyxy:
            x1,y1,x2,y2,conf,cid = row
            if conf < conf_thresh: continue
            dets.append({
                "class_id": int(cid),
                "class_name": CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES) else str(int(cid)),
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
        return dets
    except Exception as e:
        log.debug("yolov5 torch hub inference failed: %s", e)

    raise RuntimeError("Unsupported model inference API")


# Accepts both keys "image" (single) and "images" (list)
@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        # get files from either key
        incoming_files = request.files.getlist("images") or request.files.getlist("image")
        if not incoming_files:
            return jsonify({"ok": False, "error": "No files provided under key 'images' or 'image'."}), 400

        conf = float(request.form.get("conf_threshold", CONF_THRESHOLD))
        iou = float(request.form.get("iou_threshold", IOU_THRESHOLD))

        results_out = []
        for file_storage in incoming_files:
            filename = file_storage.filename or "upload"
            # verify it's an image
            try:
                image = Image.open(file_storage.stream).convert("RGB")
            except Exception as e:
                log.warning("Failed to open image %s: %s", filename, e)
                results_out.append({
                    "filename": filename,
                    "detections": [],
                    "summary": {"total": 0, "by_class": {}},
                    "image_base64": None
                })
                continue

            detections = run_inference_pil(image, conf_thresh=conf, iou_thresh=iou)

            # summary
            total = len(detections)
            by_class = Counter([d["class_name"] for d in detections])
            by_class_dict = dict(by_class)

            # annotated image
            annotated = image.copy()
            annotated = draw_detections(annotated, detections)
            

            results_out.append({
                "filename": filename,
                "detections": detections,
                "summary": {"total": total, "by_class": by_class_dict},
                "image_base64": data_uri
            })

        return jsonify({"ok": True, "results": results_out})
    except Exception as e:
        log.exception("Detection error")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "model_loaded": MODEL is not None}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
