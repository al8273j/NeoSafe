import pickle  
from flask import Flask, jsonify, render_template, request
import numpy as np

app = Flask(__name__)
# Load models
with open('models/GBDT_model.pkl', 'rb') as file:
    gbdt_model = pickle.load(file)

with open('models/rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

@app.route('/')
def home():
    # Default values for initial display in both cards
    data = {
        'age': 30,
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'bs': 100,
        'body_temp': 98.6,
        'heart_rate': 75
    }
    fetal_data = {
        'baseline_value': 120,
        'accelerations': 2.0,
        'fetal_movement': 5.0,
        'uterine_contractions': 3.0,
        'light_decelerations': 1.0,
        'severe_decelerations': 0.0,
        'prolongued_decelerations': 0.0,
        'abnormal_short_term_variability': 0.5,
        'mean_value_of_short_term_variability': 6.0,
        'percentage_of_time_with_abnormal_long_term_variability': 15.0,
        'mean_value_of_long_term_variability': 25.0
    }
    return render_template('index.html', data=data, fetal_data=fetal_data, maternal_risk=None, fetal_risk=None)

@app.route('/predict_maternal', methods=['POST'])
def predict_maternal():
    # Retrieve previous fetal data for display
    fetal_data = {
        'baseline_value': float(request.form.get('baseline_value', 120)),
        'accelerations': float(request.form.get('accelerations', 2.0)),
        'fetal_movement': float(request.form.get('fetal_movement', 5.0)),
        'uterine_contractions': float(request.form.get('uterine_contractions', 3.0)),
        'light_decelerations': float(request.form.get('light_decelerations', 1.0)),
        'severe_decelerations': float(request.form.get('severe_decelerations', 0.0)),
        'prolongued_decelerations': float(request.form.get('prolongued_decelerations', 0.0)),
        'abnormal_short_term_variability': float(request.form.get('abnormal_short_term_variability', 0.5)),
        'mean_value_of_short_term_variability': float(request.form.get('mean_value_of_short_term_variability', 6.0)),
        'percentage_of_time_with_abnormal_long_term_variability': float(request.form.get('percentage_of_time_with_abnormal_long_term_variability', 15.0)),
        'mean_value_of_long_term_variability': float(request.form.get('mean_value_of_long_term_variability', 25.0))
    }
    
    try:
        # Retrieve and parse maternal form data
        data = {
            'age': float(request.form['age']),
            'systolic_bp': float(request.form['systolicBP']),
            'diastolic_bp': float(request.form['diastolicBP']),
            'bs': float(request.form['bs']),
            'body_temp': float(request.form['bodyTemp']),
            'heart_rate': float(request.form['heartRate'])
        }
        
        # Predict maternal risk using the GBDT model
        maternal_prediction = gbdt_model.predict([[data['age'], data['systolic_bp'], data['diastolic_bp'], data['bs'], data['body_temp'], data['heart_rate']]])
        maternal_risk = "High Risk" if maternal_prediction[0] == "high hisk" else ('Mid Risk' if maternal_prediction[0] == "mid risk" else 'Low Risk')
    except (ValueError, KeyError):
        maternal_risk = "Please complete all fields for maternal risk prediction."
        data = None
        

    return render_template('index.html', data=data, fetal_data=fetal_data, maternal_risk=maternal_risk, fetal_risk=None)

@app.route('/predict_fetal', methods=['POST'])
def predict_fetal():
    # Retrieve previous maternal data for display
    data = {
        'age': float(request.form.get('age', 30)),
        'systolic_bp': float(request.form.get('systolicBP', 120)),
        'diastolic_bp': float(request.form.get('diastolicBP', 80)),
        'bs': float(request.form.get('bs', 100)),
        'body_temp': float(request.form.get('bodyTemp', 98.6)),
        'heart_rate': float(request.form.get('heartRate', 75))
    }

    try:
        # Retrieve and parse fetal form data
        fetal_data = {
            'baseline_value': float(request.form['baseline_value']),
            'accelerations': float(request.form['accelerations']),
            'fetal_movement': float(request.form['fetal_movement']),
            'uterine_contractions': float(request.form['uterine_contractions']),
            'light_decelerations': float(request.form['light_decelerations']),
            'severe_decelerations': float(request.form['severe_decelerations']),
            'prolongued_decelerations': float(request.form['prolongued_decelerations']),
            'abnormal_short_term_variability': float(request.form['abnormal_short_term_variability']),
            'mean_value_of_short_term_variability': float(request.form['mean_value_of_short_term_variability']),
            'percentage_of_time_with_abnormal_long_term_variability': float(request.form['percentage_of_time_with_abnormal_long_term_variability']),
            'mean_value_of_long_term_variability': float(request.form['mean_value_of_long_term_variability'])
        }

        # Predict fetal risk using the RF model
        fetal_prediction = rf_model.predict([[fetal_data[key] for key in fetal_data]])
        fetal_risk = 'Normal' if fetal_prediction[0] == 0 else ('Suspect' if fetal_prediction[0] == 1 else 'Pathological')
    except (ValueError, KeyError):
        fetal_risk = "Please complete all fields for fetal risk prediction."
        fetal_data = None

    return render_template('index.html', data=data, fetal_data=fetal_data, maternal_risk=None, fetal_risk=fetal_risk)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)  # Listen on all interfaces

