# Window OS
# venv\Scripts\activate
# flask run
# http://127.0.0.1:5000

# Mac OS
# source venv/bin/activate
# flask run
# http://127.0.0.1:8000/

# Update on training model
# Existing training data (https://www.kaggle.com/datasets/mirichoi0218/insurance)
# python train_model.py


import csv
import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Load the trained XGBoost model if it exists
MODEL_PATH = 'model.pkl'

@app.route('/')
def home():
    return render_template('index.html', name="Coding partner")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save and handle upload if needed
            file.save("insurance_data.csv")
            return redirect(url_for('home'))

    return render_template('upload.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    
    if request.method == 'POST':
        if not os.path.exists('model.pkl') or not os.path.exists('model_columns.pkl'):
            return "Model files not found! Please train the model first.", 400
        
        model = joblib.load('model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        
        # Get all input values from the form
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # 2. Save to CSV
        user_data = [age, sex, bmi, children, smoker, region, 0] # 0 is a placeholder for 'charges'
        with open('user_submissions.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header only if file is new
            if os.path.getsize('user_submissions.csv') == 0:
                writer.writerow(['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'])
            writer.writerow(user_data)
            
        
        # Create a DataFrame with the user input
        input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                                  columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        
        # Apply one-hot encoding to match training data structure
        input_encoded = pd.get_dummies(input_data, columns=['sex', 'smoker', 'region'])
        
        # Reindex to ensure all columns match the trained model (fill missing dummy columns with 0)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Run prediction
        prediction = model.predict(input_encoded)[0]
        prediction_result = f"Estimated Insurance Charges: ${prediction:,.2f}"
        
    return render_template('predict.html', prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(port=8000, debug=True)