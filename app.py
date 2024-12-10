from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///housing_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model to store input data
class HousingData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    longitude = db.Column(db.Float, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    housing_median_age = db.Column(db.Float, nullable=False)
    total_rooms = db.Column(db.Float, nullable=False)
    total_bedrooms = db.Column(db.Float, nullable=False)
    population = db.Column(db.Float, nullable=False)
    households = db.Column(db.Float, nullable=False)
    median_income = db.Column(db.Float, nullable=False)
    ocean_proximity_NEAR_BAY = db.Column(db.Integer, nullable=False)
    ocean_proximity_INLAND = db.Column(db.Integer, nullable=False)
    ocean_proximity_NEAR_OCEAN = db.Column(db.Integer, nullable=False)
    ocean_proximity_ISLAND = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<HousingData {self.id}>'

# Create DB if it doesn't exist
with app.app_context():
    db.create_all()

# Load and preprocess data for model
data = pd.read_csv('Intern Housing Data India.csv')
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].mean())
data.dropna(thresh=9, inplace=True)

# Prepare features for model
data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)
X = data.drop(columns=['median_house_value'])
y = data['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Page 1: Descriptive Stats
@app.route('/descriptive_stats')
def descriptive_stats():
    desc_stats = data.describe().transpose()
    return render_template('descriptive_stats.html', stats=desc_stats.to_html())

# Page 2: Inferential Stats
@app.route('/inferential_stats')
def inferential_stats():
    corr_matrix = data.corr()
    return render_template('inferential_stats.html', corr_matrix=corr_matrix.to_html())

# Page 3: Model Prediction
@app.route('/model_prediction', methods=['GET', 'POST'])
def model_prediction():
    if request.method == 'POST':
        input_data = {
            "longitude": float(request.form['longitude']),
            "latitude": float(request.form['latitude']),
            "housing_median_age": float(request.form['housing_median_age']),
            "total_rooms": float(request.form['total_rooms']),
            "total_bedrooms": float(request.form['total_bedrooms']),
            "population": float(request.form['population']),
            "households": float(request.form['households']),
            "median_income": float(request.form['median_income']),
            "ocean_proximity_NEAR_BAY": int(request.form['ocean_proximity_NEAR_BAY']),
            "ocean_proximity_INLAND": int(request.form['ocean_proximity_INLAND']),
            "ocean_proximity_NEAR_OCEAN": int(request.form['ocean_proximity_NEAR_OCEAN']),
            "ocean_proximity_ISLAND": int(request.form['ocean_proximity_ISLAND']),
        }

        # Convert input data into DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        # Align columns with the training data (X_train)
        for col in X_train.columns:
            if col not in input_df:
                input_df[col] = 0  # Add missing columns with default value 0
        input_df = input_df[X_train.columns]  # Ensure column order matches training data

        # Make prediction
        prediction = rf.predict(input_df)
        return render_template('model_prediction.html', prediction=prediction[0])

    return render_template('model_prediction.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
