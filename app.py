
import os
import uuid
from flask import Flask, render_template, request, redirect
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Load YOLO once
model = YOLO("models/best.pt")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    # Save uploaded file
    file_id = uuid.uuid4().hex
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{file_id}.jpg")
    file.save(input_path)

    # Run YOLO prediction (saves results in static/outputs/<file_id>/)
    results = model.predict(
        source=input_path,
        save=True,
        project=app.config["OUTPUT_FOLDER"],
        name=file_id
    )

    # Get predicted image filename
    output_dir = os.path.join(app.config["OUTPUT_FOLDER"], file_id)
    predicted_file = None
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.jpg', '.png'))]
        if files:
            predicted_file = files[0]

    # Process detections for table
    detections_list = []
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            det = {
                "class_name": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 2),
                "box": [int(x) for x in box.xyxy[0]]
            }
            detections_list.append(det)

    return render_template(
        "result.html",
        uploaded_image=os.path.basename(input_path),
        predicted_image=predicted_file,
        output_folder=file_id,
        detections=detections_list
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
