from flask import Flask, request, jsonify, render_template,url_for,app
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the KNN model and scaler
knnmodel = pickle.load(open('knnmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Define your column names before preprocessing
input_columns = [
    'Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 
    'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'DailyRate', 
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 
    'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
    'YearsWithCurrManager'
]

# Define your column names after preprocessing (one-hot encoding)
encoded_columns = [
    'Age', 'BusinessTravel_Travel_Rarely', 'BusinessTravel_Travel_Frequently',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical',
    'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Male',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager',
    'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative', 'MaritalStatus_Married',
    'MaritalStatus_Single', 'OverTime_Yes',
    'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']  # Access JSON data
        
        # Convert data to DataFrame
        df = pd.DataFrame([data], columns=input_columns)
        
        # Apply one-hot encoding
        df_encoded = pd.get_dummies(df).reindex(columns=encoded_columns, fill_value=0)
        
        # Transform using the scaler
        new_data_scaled = scalar.transform(df_encoded)
        
        # Make prediction using KNN model
        prediction = knnmodel.predict(new_data_scaled)
        
        # Return prediction as JSON response
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = {col: request.form.get(col) for col in input_columns}
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Apply one-hot encoding
        #df_encoded = pd.get_dummies(df).reindex(columns=encoded_columns, fill_value=0)
        
        # Transform using the scaler
        new_data_scaled = scalar.transform(df)
        
        # Make prediction using KNN model
        prediction = knnmodel.predict(new_data_scaled)[0]
        
        # Render the result in HTML template
        return render_template("home.html", predicted_text="Is there risk of Attrition: {}".format(prediction))
    
    except Exception as e:
        return render_template("home.html", predicted_text="Error: {}".format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
