import pickle
import numpy as np
import pandas as pd
import json
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')
api_url = 'http://localhost:5000/stroke'
filename = 'logistic_regression_stroke.sav'
loaded_model = pickle.load(open(filename, 'rb'))
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    BMI = float(request.form['BMI'])

    data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "BMI": BMI
    }
    features_list = list(data.values())
    prediction = str(loaded_model.predict([features_list]))
    if(prediction=="[0]"): prediction = "[0] - Non-stroke"
    if(prediction=="[1]"): prediction = "[1] - Stroke"
    confidence = str(loaded_model.predict_proba([features_list]))
    c1,c2 = confidence.split(" ")
    c1 = float(c1[2:])
    c1 = round(c1,2)
    c2 = c2[:len(c2)-2]
    return render_template('result.html', prediction=prediction, confidence=confidence,c1=c1,c2=c2)
if __name__ == '__main__':
    app.run()