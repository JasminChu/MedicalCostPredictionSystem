# Flask app for Vercel (zero-config): expose a top-level variable named `app`.
# Local run:  flask run   (or: python app.py)

# Window OS
# venv\Scripts\activate
# flask run
# http://127.0.0.1:5000

# Mac OS
# source venv/bin/activate
# flask run
# http://127.0.0.1:8000/
# uv run app.py  

# Update on training model
# Existing training data (https://www.kaggle.com/datasets/mirichoi0218/insurance)
# python train_model.py

import csv
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# templates/ and static/ sit next to this file, so Flask's defaults work.
app = Flask(__name__)

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

# Vercel's filesystem is read-only except /tmp (and /tmp is not persistent).
if os.environ.get("VERCEL"):
    UPLOAD_FOLDER = "/tmp/uploads"
else:
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
SUBMISSIONS_CSV = os.path.join(UPLOAD_FOLDER, "user_submissions.csv")

# Load the model once per cold start, not on every request.
_model = None
_model_columns = None


def get_model():
    global _model, _model_columns
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _model_columns = joblib.load(COLUMNS_PATH)
    return _model, _model_columns


@app.route("/")
def home():
    return render_template("index.html", name="Coding partner")


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    message = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part in the request", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        if file.filename.lower().endswith(".csv"):
            file.save(SUBMISSIONS_CSV)
            message = "File successfully uploaded! You can now retrain your model."
        else:
            message = "Invalid file type. Please upload a .csv file."

    return render_template("upload.html", message=message)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_result = None

    if request.method == "POST":
        if not (os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH)):
            return "Model files not found! Please train the model first.", 400

        model, model_columns = get_model()

        age = float(request.form["age"])
        sex = request.form["sex"]
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]

        # Log the submission (charges=0 is a placeholder). Wrapped so a
        # filesystem problem can never break the prediction itself.
        try:
            is_new = (not os.path.exists(SUBMISSIONS_CSV)
                      or os.path.getsize(SUBMISSIONS_CSV) == 0)
            with open(SUBMISSIONS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(["age", "sex", "bmi", "children",
                                     "smoker", "region", "charges"])
                writer.writerow([age, sex, bmi, children, smoker, region, 0])
        except OSError as e:
            app.logger.warning("Could not save submission: %s", e)

        input_data = pd.DataFrame(
            [[age, sex, bmi, children, smoker, region]],
            columns=["age", "sex", "bmi", "children", "smoker", "region"],
        )
        input_encoded = pd.get_dummies(input_data, columns=["sex", "smoker", "region"])
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]
        prediction_result = f"Estimated Insurance Charges: ${prediction:,.2f}"

    return render_template("predict.html", prediction_result=prediction_result)


if __name__ == "__main__":
    app.run(port=1234, debug=True)
