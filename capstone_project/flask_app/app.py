import logging
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request

PREPROCESSOR_PATH='capstone_project/models/power_transformer.pkl'
MODEL_PATH='capstone_project/models/model.pkl'

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

app=Flask(__name__)

def load_model(path_name:str):
    with open(path_name,'rb') as file:
        model=pickle.load(file)
    return model

def load_transformer(path_name:str):
    with open(path_name,'rb') as file:
        transformer=pickle.load(file)
    return transformer

model=load_model(MODEL_PATH)
transformer=load_transformer(PREPROCESSOR_PATH)

def preprocess_input(data):
    ip_data=np.array(data).reshape(1,-1)
    transformed_data=transformer.transform(ip_data)
    return transformed_data

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    input_values = [""] * len(FEATURE_NAMES)

    if request.method == "POST":
        csv_input = request.form.get("csv_input", "").strip()
        if csv_input:
            try:
                values = list(map(float, csv_input.split(",")))
                if len(values) != len(FEATURE_NAMES):
                    raise ValueError(f"Expected {len(FEATURE_NAMES)} values, got {len(values)}")

                input_values = values
                transformed_features = preprocess_input(values)
                if transformed_features is not None and model:
                    result = model.predict(transformed_features)
                    prediction = "Fraud" if result[0] == 1 else "Non-Fraud"
                else:
                    prediction = "Error: Model or Transformer not loaded properly."
            except Exception as e:
                prediction = f"Error: {e}"

    return render_template("index.html", result=prediction, csv_input=",".join(map(str, input_values)))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


