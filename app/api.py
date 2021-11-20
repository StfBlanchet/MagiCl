#!/usr/bin/env python
# coding: utf8

"""
Flask API
"""

import json

from app import app
from flask import jsonify, request

from .Preprocessor import Preprocessor
from .Modeler import Modeler
from .Classifier import Classifier


@app.route("/", methods=["GET"])
def hello():
    quick_user_guide = open("quick_user_guide.html", "r")
    return quick_user_guide.read()


@app.route("/preprocess/<directory>/<filename>", methods=["GET"])
def preprocess(directory, filename):
    params = dict(request.args)
    data = Preprocessor(directory, filename, params).run()

    return json.loads(data.to_json())


@app.route("/model/<directory>/<filename>", methods=["GET"])
def model(directory, filename):
    params = dict(request.args)
    data = Modeler(directory, filename, params).run()

    return jsonify(data)


@app.route("/classify/<directory>/<filename>", methods=["GET"])
def classify(directory, filename):
    data = Classifier(directory, filename).run()

    return json.loads(data)
