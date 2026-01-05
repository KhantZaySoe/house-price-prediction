import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load trained model and feature columns
model = joblib.load("condo_price_model.pkl")
TRAINING_COLUMNS = joblib.load("model_columns.pkl")

# Load training dataset stats (mean values for missing features)
# ⚠️ Make sure you have saved X_train_mean.pkl during training
try:
    FEATURE_MEANS = joblib.load("X_train_mean.pkl")
except:
    FEATURE_MEANS = {col: 0 for col in TRAINING_COLUMNS}  # fallback

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json()

        # Start with mean values (instead of 0)
        input_data = {col: FEATURE_MEANS.get(col, 0) for col in TRAINING_COLUMNS}

        # Update with user inputs
        if "area" in json_data and json_data["area"]:
            input_data["Total_Area_squft"] = float(json_data["area"])

        if "bedrooms" in json_data and json_data["bedrooms"]:
            input_data["Bedrooms"] = int(json_data["bedrooms"])

        if "bathrooms" in json_data and json_data["bathrooms"]:
            input_data["Bathrooms"] = int(json_data["bathrooms"])

        if "township" in json_data and json_data["township"]:
            township = json_data["township"]
            if township in input_data:
                input_data[township] = 1

        if "amenities" in json_data:
            total_amenities = sum(json_data["amenities"].values())
            input_data["Total_Amenities"] = total_amenities

        # Convert into DataFrame
        final_data = pd.DataFrame([input_data], columns=TRAINING_COLUMNS)

        # Predict
        prediction_log = model.predict(final_data)
        prediction = np.expm1(prediction_log)

        usd_price = round(prediction[0])
        mmk_price = usd_price * 2100  # conversion

        return jsonify({
            "predicted_price_usd": usd_price,
            "predicted_price_mmk": mmk_price
        })

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
