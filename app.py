"""Flask app that trains a simple regressor on the California housing dataset
and exposes a web form and JSON API to predict housing price from user input.

Run:
    python app.py

Then open http://127.0.0.1:5000/ in a browser.
"""
from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

app = Flask(__name__)
MODEL_PATH = "price_model.joblib"
CSV_PATH = "house_price_regression_dataset.csv"

FEATURE_NAMES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]


def train_and_save_model():
    # If model exists, load it
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        return model

    # Load dataset from CSV
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    X = df[FEATURE_NAMES].values
    y = df["House_Price"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"Trained model MAE: ${mae:,.2f}")

    joblib.dump(model, MODEL_PATH)
    return model


model = train_and_save_model()


def parse_features_from_dict(d):
    """Return a numpy array shaped (1, 7) from a dict-like input.
    The numeric fields are expected; missing values will raise a ValueError.
    """
    vals = []
    for name in FEATURE_NAMES:
        if name not in d:
            raise ValueError(f"Missing feature: {name}")
        vals.append(float(d[name]))
    return np.array(vals, dtype=float).reshape(1, -1)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    # Default values based on dataset median-like ranges
    defaults = dict(zip(FEATURE_NAMES, [2500, 3, 2, 2000, 3.0, 1, 6]))

    if request.method == "POST":
        try:
            # collect form fields
            inputs = {name: request.form[name] for name in FEATURE_NAMES}
            x = parse_features_from_dict(inputs)
            pred = model.predict(x)[0]
            result = f"Estimated price: ${pred:,.2f}"
        except Exception as e:
            error = str(e)

    return render_template("index.html", feature_names=FEATURE_NAMES, defaults=defaults, result=result, error=error)


@app.route("/api/predict", methods=["POST", "GET"])
def api_predict():
    """Accepts JSON (POST) or query params (GET) with feature names and returns predicted price in USD."""
    # Support POST with JSON (preferred) and GET with query params (convenience)
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({"error": "Expected JSON body for POST"}), 400
        data = request.get_json()
    else:
        # GET: read features from query string
        data = request.args.to_dict()

    try:
        x = parse_features_from_dict(data)
        pred = model.predict(x)[0]
        price = float(pred)
        return jsonify({"predicted_price": price, "units": "USD"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Helpful convenience route: allow `/index.html` to map to the main page
@app.route('/index.html')
def index_html_redirect():
    return redirect(url_for('index'))


# Serve favicon from static if present, otherwise return 204 (no content)
@app.route('/favicon.ico')
def favicon():
    static_folder = os.path.join(app.root_path, 'static')
    favicon_path = os.path.join(static_folder, 'favicon.ico')
    if os.path.exists(favicon_path):
        return send_from_directory(static_folder, 'favicon.ico')
    # no favicon available; return no content so clients stop retrying 404s
    return ('', 204)


if __name__ == "__main__":
    # Start Flask dev server
    app.run(debug=True)
