from flask import Flask
from flask import request
import requests
import json
from flask_cors import CORS, cross_origin
from flask import jsonify

from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# Load the dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Train R2 score: {train_score:.2f}")
print(f"Test R2 score: {test_score:.2f}")

# Save the trained model to a file
joblib.dump(model, 'california_house_price_model.pkl')

# Load the trained model
model = joblib.load('california_house_price_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict_price():
    if request.method == 'POST':
        # Get user inputs for the features
        input_features = [float(request.form['MedInc']),
                          float(request.form['HouseAge']),
                          float(request.form['AveRooms']),
                          float(request.form['AveBedrms']),
                          float(request.form['Population']),
                          float(request.form['AveOccup']),
                          float(request.form['Latitude']),
                          float(request.form['Longitude'])]

        # Scale the user inputs using the same scaler used during training
        scaled_features = scaler.transform([input_features])

        # Make a prediction using the model
        predicted_price = model.predict(scaled_features)[0]

        predict_price.json()

    return predict_price.json()

if __name__ == '__main__':
    app.run(debug=False)