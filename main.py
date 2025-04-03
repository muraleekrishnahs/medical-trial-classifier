from flask import Flask, jsonify, request
from typing import Literal
from src.model import MedicalTrialClassifier
import torch
import os


app = Flask(__name__)


LABELS = Literal[
    "Dementia",
    "ALS",
    "Obsessive Compulsive Disorder",
    "Scoliosis",
    "Parkinson's Disease",
]

# Load the trained model
model = MedicalTrialClassifier()
model_path = os.path.join('src', 'best_model.pt')
model.load(model_path)


def predict(description: str) -> LABELS:
    """
    Function that should take in the description text and return the prediction
    for the class that we identify it to.
    The possible classes are: ['Dementia', 'ALS',
                                'Obsessive Compulsive Disorder',
                                'Scoliosis', 'Parkinson's Disease']
    """
    return model.predict(description)


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def identify_condition():
    try:
        data = request.get_json(force=True)
        prediction = predict(data["description"])
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run()