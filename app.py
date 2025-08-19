import os
import shutil
import uuid
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO

# Initialize Flask
app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Load YOLO model once
model = YOLO("models/best.pt")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        # Save uploaded file
        file_id = uuid.uuid4().hex
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.jpg")
        output_dir = os.path.join(app.config["OUTPUT_FOLDER"], file_id)
        os.makedirs(output_dir, exist_ok=True)
        file.save(input_path)

        # Run YOLO prediction (save inside unique subdir)
        results = model.predict(
            source=input_path,
            save=True,
            project=app.config["OUTPUT_FOLDER"],
            name=file_id
        )

        # YOLO saves to outputs/fileid/input.jpg -> copy it to flat outputs
        pred_path = os.path.join(output_dir, os.path.basename(input_path))
        final_output = os.path.join(app.config["OUTPUT_FOLDER"], f"{file_id}_pred.jpg")
        if os.path.exists(pred_path):
            shutil.copy(pred_path, final_output)

        return render_template(
            "result.html",
            input_image=url_for("static", filename=f"uploads/{os.path.basename(input_path)}"),
            output_image=url_for("static", filename=f"outputs/{os.path.basename(final_output)}"),
            detections=results[0].boxes.data.tolist() if results else []
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
