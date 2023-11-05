# API
from flask import Flask, request, jsonify
from flask_cors import CORS
import csv

# ML
import joblib
import numpy as np

NAME_KEC = 1
BED_LIMIT = 2
POPULATION = 3
NON_ISPA_PATIENT = 2

NON_ISPA_PATIENT_COUNT_AVERAGE = 593
JUMLAH_PENDUDUK_JAKARTA = 10344018

db_kecamatan = {}
with open('bed_population.csv', mode='r', encoding='UTF8') as f:
    # reading the CSV file
    csvFile = csv.reader(f)

    # displaying the contents of the CSV file
    for lines in csvFile:
        db_kecamatan[lines[NAME_KEC]]={
            "name_kec":lines[NAME_KEC],
            "bed_limit":lines[BED_LIMIT],
            "non_ispa_patient":lines[NON_ISPA_PATIENT],
            "count_penduduk":int(lines[POPULATION]),
        }

app = Flask(__name__)
CORS(app)

# # Load the trained model
model = joblib.load('xgboost.pkl')

@app.route("/")
def home():
    return "HTTP 200 OK"

@app.route('/api', methods=['GET'])
def predict_price():
    # Get user inputs for the features
    input_data = {
        "pm10": float(request.args.get('pm10')),
        "so2": float(request.args.get('so2')),
        "co": float(request.args.get('co')),
        "o3": float(request.args.get('o3')),
        "no2": float(request.args.get('no2'))
    }

    # Ensure the input data has the required features
    required_features = ['pm10', 'so2', 'co', 'o3', 'no2']
    if not all(feature in input_data for feature in required_features):
        return jsonify({'error': 'Missing required features'}), 400

    # Prepare the input data as a NumPy array for prediction
    features = np.array([[input_data[feature] for feature in required_features]])

    # Make predictions using the loaded model
    prediction = model.predict(features)

    # Total ISPA patient
    total_patient_prediction = float(prediction[0])
    data = []
    for k,v in db_kecamatan.items():
        data.append({
            "nama_kec":v["name_kec"],
            "total_bed":v["bed_limit"],
            "hexcode":get_kecamatan_color(v["name_kec"],total_patient_prediction)
        },)

    # Return the prediction as JSON
    # return jsonify({'prediction': float(prediction[0])})


    return jsonify(data)

def get_kecamatan_color(name_kec, total_patient_prediction):
    if int(db_kecamatan[name_kec]["bed_limit"]) == 0:
        return "red"
    ratio_penduduk = db_kecamatan[name_kec]["count_penduduk"] / JUMLAH_PENDUDUK_JAKARTA
    ispa_bed_available = int(db_kecamatan[name_kec]["bed_limit"]) - (NON_ISPA_PATIENT_COUNT_AVERAGE * ratio_penduduk) - (total_patient_prediction * ratio_penduduk)
    if ispa_bed_available <0:
        return "maroon"
    bed_remaining_percentage = ispa_bed_available / int(db_kecamatan[name_kec]["bed_limit"])
    print("============")
    print("name_kec", name_kec)
    print("total_patient_prediction_jakarta", total_patient_prediction)
    print("total_patient_prediction_kecamatan", total_patient_prediction * ratio_penduduk)
    print("penduduk", db_kecamatan[name_kec]["count_penduduk"])
    print("bed_limit", db_kecamatan[name_kec]["bed_limit"])
    print("ispa_bed_available", ispa_bed_available)
    print("bed_remaining_percentage", bed_remaining_percentage)
    print("============")
    if bed_remaining_percentage <= 0.3:
        return "green"
    elif bed_remaining_percentage <= 0.70:
        return "yellow"
    elif bed_remaining_percentage <= 1:
        return "red"
    else:
        return "maroon"

if __name__ == '__main__':
    app.run(debug=False)

