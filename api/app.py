from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline (preprocessing + model bundled together)
model = joblib.load("model.pkl")

EXPECTED_FIELDS = ["from", "to", "flightType", "agency", "distance", "month", "day_of_week"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON body provided"}), 400

    missing = [f for f in EXPECTED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        input_df = pd.DataFrame([{field: data[field] for field in EXPECTED_FIELDS}])
        prediction = model.predict(input_df)[0]
        return jsonify({"predicted_price": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
