from pathlib import Path
from math import isfinite

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import BadRequest

FEATURES = [
    "Tavg",
    "Prcp",
    "Wspd",
    "Humidity",
    "Cloud_Cover",
    "Tmin",
    "Tmax",
    "Dew_Point",
]
MAX_PLAY_TEMPERATURE = 35
MAX_PLAY_PRECIPITATION = 5
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

MODEL_PATH = BASE_DIR / "weather_knn_model.pkl"
DATA_PATH = BASE_DIR / "weather_feature_engineered.csv"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. Run KNN.py to train and save it."
    )

model = joblib.load(MODEL_PATH)
model_features = getattr(model, "feature_names_in_", None)
if model_features is not None and list(model_features) != FEATURES:
    raise ValueError(
        "The saved model was trained with different input features. "
        "Run KNN.py again to retrain and replace weather_knn_model.pkl."
    )

if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Weather feature data not found at {DATA_PATH}.")

weather = pd.read_csv(DATA_PATH)
missing_features = [feature for feature in FEATURES if feature not in weather.columns]
if missing_features:
    raise ValueError(
        f"Weather feature data is missing required columns: {missing_features}"
    )

FEATURE_RANGES = {
    feature: [float(weather[feature].min()), float(weather[feature].max())]
    for feature in FEATURES
}


@app.get("/")
def home():
    return send_file(BASE_DIR / "webpage.html")


@app.get("/ranges")
def ranges():
    return jsonify(FEATURE_RANGES)


@app.post("/predict")
def predict():
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 415

    try:
        values = request.get_json()
    except BadRequest:
        return jsonify({"error": "Request body contains invalid JSON."}), 400

    if not isinstance(values, dict):
        return jsonify({"error": "JSON body must be an object of feature values."}), 400

    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        return jsonify({"error": "Missing required features.", "features": missing}), 400

    unknown = [feature for feature in values if feature not in FEATURES]
    if unknown:
        return jsonify({"error": "Unknown features.", "features": unknown}), 400

    numeric_values = {}
    for feature in FEATURES:
        value = values[feature]
        if isinstance(value, bool):
            return jsonify({"error": f"{feature} must be a finite number."}), 400

        try:
            number = float(value)
        except (TypeError, ValueError):
            return jsonify({"error": f"{feature} must be a finite number."}), 400

        if not isfinite(number):
            return jsonify({"error": f"{feature} must be a finite number."}), 400

        numeric_values[feature] = number

    advice = []
    too_hot = numeric_values["Tmax"] > MAX_PLAY_TEMPERATURE
    heavy_rain = numeric_values["Prcp"] > MAX_PLAY_PRECIPITATION

    if too_hot:
        advice.append("Too hot to play safely. Postpone the game and stay hydrated.")
    if heavy_rain:
        advice.append("Wet conditions may make the ground slippery. Postpone the game.")

    if too_hot or heavy_rain:
        football_play = "No"
    else:
        input_data = pd.DataFrame([numeric_values], columns=FEATURES)
        predicted_class = model.predict(input_data)[0]
        football_play = "Yes" if predicted_class == 1 else "No"

    return jsonify({"football_play": football_play, "advice": advice})


if __name__ == "__main__":
    app.run(debug=True)
