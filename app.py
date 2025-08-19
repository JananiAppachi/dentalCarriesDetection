import os
from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import uuid

# Initialize Flask
app = Flask(__name__)

# Paths
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Load YOLO model once (make sure models/best.pt exists in repo)
model = YOLO("models/best.pt")

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        return redirect(request.url)

    if file:
        # Save uploaded image
        file_id = str(uuid.uuid4().hex)  # unique ID
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], file_id + ".jpg")
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], file_id + "_pred.jpg")
        file.save(input_path)

        # Run YOLO prediction
        results = model.predict(source=input_path, save=True, project=app.config["OUTPUT_FOLDER"], name=file_id)

        # YOLO saves result in a subfolder -> move/rename output
        pred_path = os.path.join(app.config["OUTPUT_FOLDER"], file_id, os.path.basename(input_path))
        if os.path.exists(pred_path):
            os.rename(pred_path, output_path)

        return render_template("result.html",
                               input_image=url_for("static", filename="uploads/" + os.path.basename(input_path)),
                               output_image=url_for("static", filename="outputs/" + os.path.basename(output_path)))

# Render entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)
