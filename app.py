import os, uuid
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2
import numpy as np

# --- Config paths ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
MODEL_DIR  = BASE_DIR / "models"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load YOLO model once ---
yolo_path = MODEL_DIR / "best.pt"
model = YOLO(str(yolo_path))

# class names will come from the weights
CLASS_NAMES = model.names  # dict like {0:'Caries', 1:'Cavity', ...}

app = Flask(__name__)

def run_yolo(input_path: Path, out_fn: str):
    """
    Run YOLO on a single image and save:
      - plotted image to OUTPUT_DIR / out_fn
      - return detections as a list of dicts
    """
    results = model.predict(
        source=str(input_path),
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )
    res = results[0]

    # Save nice plotted image
    plotted = res.plot()  # np array BGR
    out_path = OUTPUT_DIR / out_fn
    cv2.imwrite(str(out_path), plotted)

    # Build detections table
    dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            cls_id = int(b.cls[0].item())
            conf  = float(b.conf[0].item())
            xyxy  = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
            dets.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                "confidence": round(conf, 3),
                "box": [round(x, 1) for x in xyxy]
            })
    return str(out_path.name), dets

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("index"))
    f = request.files["image"]
    if f.filename == "":
        return redirect(url_for("index"))

    # save upload
    ext = os.path.splitext(f.filename)[1].lower()
    uid = uuid.uuid4().hex[:10]
    in_name = f"{uid}{ext if ext in ['.jpg','.jpeg','.png'] else '.jpg'}"
    in_path = UPLOAD_DIR / in_name
    f.save(str(in_path))

    # run yolo
    out_name = f"{uid}_pred.jpg"
    out_img_name, detections = run_yolo(in_path, out_name)

    return render_template(
        "result.html",
        uploaded_image=in_path.name,
        predicted_image=out_img_name,
        detections=detections
    )

if __name__ == "__main__":
    # For local demo
    app.run(host="127.0.0.1", port=5000, debug=True)
