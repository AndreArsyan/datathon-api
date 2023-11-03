# API
from flask import Flask, request, jsonify
from flask_cors import CORS

# ML
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# # Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform([[1,2,3,4,5,6,7,8]])
X_test_scaled = scaler.transform([[1,2,3,4,5,6,7,8]])

# # Load the trained model
model = joblib.load('california_house_price_model.pkl')

@app.route("/")
def home():
    return "HTTP 200 OK"

@app.route('/api', methods=['GET'])
def predict_price():
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


if __name__ == '__main__':
    app.run(debug=False)

