from flask import Flask
from flask import request
# import requests
# import json
from flask_cors import CORS, cross_origin
from flask import jsonify

from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import pickle
import numpy as np

from app2 import app2_api

app = Flask(__name__)
app.register_blueprint(app2_api)
CORS(app)
#
# Load the dataset
california = fetch_california_housing()
X = california.data
y = california.target
#
# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # Load the trained model
model = joblib.load('california_house_price_model.pkl')

@app.route("/")
def home():
    return "HTTP 200 OK"

@app.route('/api', methods=['GET', 'POST'])
def predict_price():
    if request.method == 'GET':
        # Get user inputs for the features
        input_features = [float(request.args.get('MedInc')),
                          float(request.args.get('HouseAge')),
                          float(request.args.get('AveRooms')),
                          float(request.args.get('AveBedrms')),
                          float(request.args.get('Population')),
                          float(request.args.get('AveOccup')),
                          float(request.args.get('Latitude')),
                          float(request.args.get('Longitude'))]

        # Scale the user inputs using the same scaler used during training
        scaled_features = scaler.transform([input_features])

        # Make a prediction using the model
        predicted_price = model.predict(scaled_features)[0]


        return jsonify(predicted_price)

        # return jsonify(0)


if __name__ == '__main__':
    app.run(debug=False)