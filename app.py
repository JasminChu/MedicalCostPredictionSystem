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
import numpy as np
import pandas as pd
import shap
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

# Load the model and SHAP explainer once per cold start, not on every request.
_model = None
_model_columns = None
_explainer = None


def get_model():
    global _model, _model_columns, _explainer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _model_columns = joblib.load(COLUMNS_PATH)
        _explainer = shap.TreeExplainer(_model)
    return _model, _model_columns, _explainer


def build_shap_contributions(model_columns, encoded_row, row_shap, raw_inputs):
    """Build readable SHAP rows, showing only the selected region."""
    contributions = []
    region_shap_sum = 0.0
    has_region_cols = False
    active_region_name = None

    for col, val, sv in zip(model_columns, encoded_row, row_shap):
        sv_float = float(sv)
        is_active = bool(float(val) > 0.5)

        if col.startswith("region_"):
            has_region_cols = True
            region_shap_sum += sv_float
            if is_active:
                active_region_name = col.replace("region_", "").capitalize()
            continue

        if col == "age":
            label, display_val = "Age", f"{raw_inputs['age']:g}"
        elif col == "bmi":
            label, display_val = "BMI", f"{raw_inputs['bmi']:.1f}"
        elif col == "children":
            label, display_val = "Children", f"{raw_inputs['children']}"
        elif col == "sex_male":
            label, display_val = "Sex", "Male" if is_active else "Female"
        elif col == "sex_female":
            label, display_val = "Sex", "Female" if is_active else "Male"
        elif col == "smoker_yes":
            label, display_val = "Smoker", "Yes" if is_active else "No"
        elif col == "smoker_no":
            label, display_val = "Smoker", "No" if is_active else "Yes"
        else:
            label, display_val = col, val

        impact = round(sv_float, 2)
        contributions.append(
            {
                "feature": label,
                "value": display_val,
                "impact": impact,
                "abs_impact": abs(impact),
            }
        )

    if has_region_cols:
        selected_region = (
            active_region_name
            if active_region_name is not None
            else str(raw_inputs["region"]).capitalize()
        )
        region_impact = round(region_shap_sum, 2)
        contributions.append(
            {
                "feature": "Region",
                "value": selected_region,
                "impact": region_impact,
                "abs_impact": abs(region_impact),
            }
        )

    contributions.sort(key=lambda item: item["abs_impact"], reverse=True)
    return contributions


@app.route("/")
def home():
    return render_template("index.html", name="to InsureAI")


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
    predicted_charge = None
    shap_contributions = None
    base_value = None

    if request.method == "POST":
        if not (os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH)):
            return "Model files not found! Please train the model first.", 400

        model, model_columns, explainer = get_model()

        age = float(request.form["age"])
        sex = request.form["sex"]
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]

        raw_inputs = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }

        # 1. Compute prediction first
        input_data = pd.DataFrame(
            [[age, sex, bmi, children, smoker, region]],
            columns=["age", "sex", "bmi", "children", "smoker", "region"],
        )
        input_encoded = pd.get_dummies(input_data, columns=["sex", "smoker", "region"])
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]
        predicted_charge = round(float(prediction), 2)
        prediction_result = f"Estimated Insurance Charges: ${predicted_charge:,.2f}"

        # 2. Compute SHAP feature contributions for this prediction
        raw_shap = explainer.shap_values(input_encoded)
        row_shap = np.asarray(raw_shap[0], dtype=float)
        expected = explainer.expected_value
        base_value = round(float(np.asarray(expected).reshape(-1)[0]), 2)

        shap_contributions = build_shap_contributions(
            model_columns,
            input_encoded.iloc[0].tolist(),
            row_shap.tolist(),
            raw_inputs,
        )

        # 3. Save the actual predicted_charge (e.g., 19789.14) to user_submissions.csv
        try:
            is_new = (
                not os.path.exists(SUBMISSIONS_CSV)
                or os.path.getsize(SUBMISSIONS_CSV) == 0
            )
            with open(SUBMISSIONS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(
                        ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
                    )
                writer.writerow(
                    [age, sex, bmi, children, smoker, region, predicted_charge]
                )
        except OSError as e:
            app.logger.warning("Could not save submission: %s", e)

    return render_template(
        "predict.html",
        prediction_result=prediction_result,
        predicted_charge=predicted_charge,
        shap_contributions=shap_contributions,
        base_value=base_value,
    )


if __name__ == "__main__":
    app.run(port=1234, debug=True)