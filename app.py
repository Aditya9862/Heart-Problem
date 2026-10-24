
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_model1.joblib"

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "thalach",
    "exang",
    "oldpeak",
]

model = None
model_error = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model_error = str(exc)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            # Read form values
            form_data = {
                "age": int(request.form["age"]),
                "sex": int(request.form["sex"]),
                "cp": int(request.form["cp"]),
                "trestbps": int(request.form["trestbps"]),
                "chol": int(request.form["chol"]),
                "thalach": int(request.form["thalach"]),
                "exang": int(request.form["exang"]),
                "oldpeak": float(request.form["oldpeak"]),
            }

            # Basic input validation
            if not 1 <= form_data["age"] <= 120:
                raise ValueError("Age must be between 1 and 120.")

            if form_data["sex"] not in (0, 1):
                raise ValueError("Invalid sex value.")

            if form_data["cp"] not in (0, 1, 2, 3):
                raise ValueError("Chest pain type must be between 0 and 3.")

            if not 0 <= form_data["trestbps"] <= 300:
                raise ValueError("Invalid resting blood pressure.")

            if not 0 <= form_data["chol"] <= 1000:
                raise ValueError("Invalid cholesterol value.")

            if not 0 <= form_data["thalach"] <= 300:
                raise ValueError("Invalid maximum heart rate.")

            if form_data["exang"] not in (0, 1):
                raise ValueError("Invalid exercise-induced angina value.")

            if not -10 <= form_data["oldpeak"] <= 20:
                raise ValueError("Invalid ST depression value.")

            if model is None:
                raise RuntimeError(
                    "The trained model could not be loaded. "
                    "Check that heart_model1.joblib exists and is compatible."
                )

            # Match the feature names and order used during training
            input_df = pd.DataFrame(
                [form_data],
                columns=FEATURES
            )

            # Generate prediction
            pred = int(model.predict(input_df)[0])

            if pred == 1:
                prediction = "Possible heart disease"
            else:
                prediction = "No heart disease detected by the model"

            # Get probability for class 1, if supported
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    probability = round(
                        float(probabilities[classes.index(1)]) * 100,
                        2
                    )

        except (ValueError, KeyError) as exc:
            error = f"Please check your inputs: {exc}"
        except Exception as exc:
            error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
        form_data=form_data,
        model_error=model_error,
    )


if __name__ == "__main__":
    app.run(debug=True)
