from flask import Flask, render_template, send_file,request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

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


def generate_plots():
    # Plot 1: Missing values bar plot
    missing_values = data.isnull().sum()
    plt.figure(figsize=(12, 6))
    missing_values.plot(kind='bar', color='blue')
    plt.title('Number of Missing Values in Each Column')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.savefig('static/missing_values.png')
    plt.close()

    # Plot 2: Box plot for outliers
    num_columns = len(data.columns)
    ncols = 3  # Number of columns in the grid layout
    nrows = -(-num_columns // ncols)  # Calculate rows, ensuring all columns fit (ceiling division)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5))
    axes = axes.flatten()

    # Generate a box plot for each column
    for i, column in enumerate(data.columns):
        data.boxplot(column=column, ax=axes[i])
        axes[i].set_title(f'Boxplot: {column}')

    # Hide any extra subplots (if columns < nrows * ncols)
    for j in range(num_columns, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('static/box_plot.png')
    plt.close()

    # Plot 3: Correlation heatmap
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, linewidths=1, cmap="viridis")
    plt.title('Correlation Heatmap')
    plt.savefig('static/correlation_heatmap.png')
    plt.close()

    # Plot 4: Geographical scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data['longitude'], data['latitude'], alpha=0.4, s=data['population'] / 100,
                          c=data['median_house_value'], cmap=plt.get_cmap("jet"))

    # Add colorbar to the scatter plot
    plt.colorbar(scatter, label='Median House Value')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographical Plot of Median House Value')

    # Save the plot to a file
    plt.savefig('static/geographical_plot.png')
    plt.close()

# Generate plots when the app starts
generate_plots()


@app.route('/visualization')
def visualization():
    return render_template(
        'visualization.html',
        missing_plot_url='/static/missing_values.png',
        box_plot_url='/static/box_plot.png',
        heatmap_url='/static/correlation_heatmap.png',
        geo_plot_url='/static/geographical_plot.png'
    )



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
