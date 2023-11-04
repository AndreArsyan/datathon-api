from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained XGBoost model
model = joblib.load('xgboost.pkl')

# Define the route for rendering the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.json['data']

        # Ensure the input data has the required features
        required_features = ['pm10', 'so2', 'co', 'o3', 'no2']
        if not all(feature in input_data for feature in required_features):
            return jsonify({'error': 'Missing required features'}), 400

        # Prepare the input data as a NumPy array for prediction
        features = np.array([[input_data[feature] for feature in required_features]])

        # Make predictions using the loaded model
        prediction = model.predict(features)

        # Return the prediction as JSON
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
