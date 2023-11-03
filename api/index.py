from flask import Flask
from flask import request
import requests
import json
from flask_cors import CORS, cross_origin
from flask import jsonify

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "HTTP 200 OK"
