import os
import pandas as pd
from flask import Flask, request, jsonify
import mlflow.pyfunc

MODEL_URI = os.getenv("MODEL_URI", "/app/models/best_model")

app = Flask(__name__)

print(f"Carregando modelo a partir de: {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.route("/predict", methods=["POST"])
def predict():
    content = request.get_json()
    data = content.get("data", [])
    df = pd.DataFrame(data)

    cols_to_drop = ['ad_id', 'xyz_campaign_id', 'fb_campaign_id', 'Approved_Conversion', 'Total_Conversion']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # mapear age e gender
    if "age" in df.columns:
        df["age"] = df["age"].replace({"30-34":1 ,"45-49":2 ,"35-39":3, "40-44":4})
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace({"M":1 ,"F":2})

    preds = model.predict(df)
    return jsonify({"predictions": preds.tolist()})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)