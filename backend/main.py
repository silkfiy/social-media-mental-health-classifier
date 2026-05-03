import pickle
import pandas as pd
import numpy as np
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('../linear_svc_1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('../scaler_1.pkl', 'rb') as transformer_file:
    transformer = pickle.load(transformer_file)
with open('../feature_list_1.pkl', 'rb') as feature_file:
    features = pickle.load(feature_file)

def load_data():
    try:
        with open('data.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        return jsonify({'error' : str(e)})

@app.route("/")
def home():
    return f'model active<br>required data:<br>&ensp;{list(features)}'

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'data not found'}), 400
        input = pd.DataFrame([data])
        if not all([input.columns[i]==features[i] for i in range(len(features))]):
            return jsonify({'error': 'columns not right'}), 400
        transformed_data = transformer.transform(input)
        predicted_value = model.predict(transformed_data)
        prediction = {'prediction' : predicted_value[0]}
        data['mental_state'] = predicted_value[0]
        try:
            with open('data.json', 'r') as f:
                file = json.load(f)
        except FileNotFoundError:
            file = []
        file.append(data)
        try:
            with open('data.json', 'w') as f:
                json.dump(file, f, indent=2)
        except Exception as e:
            return jsonify({'error' : str(e)})
        finally:
            file = []
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error' : f'exception: {str(e)}'})

@app.route("/test1", methods=['POST'])
def test1():
    try:
        data = request.json
        transformed_data = transformer.transform(pd.DataFrame([data]))
        return jsonify(transformed_data)
    except Exception as e:
        return jsonify({'error' : str(e)})
    
@app.route("/data", methods=['GET'])
def data():
    loaded_data = load_data()
    return jsonify(loaded_data)