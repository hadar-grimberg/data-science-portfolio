# -*- coding: utf-8 -*-
"""
Hadar Grimberg
10/2/2021

"""

from flask import Flask, jsonify, request, make_response
import pandas as pd
import numpy as np
import sklearn
import pickle
import json
from configs import cols
import sys
sys.path.insert(-1, '../code')
from model import titanicModel



clf_path = '../models/TitanicClassifier.pkl'
with open(clf_path, 'rb') as f:
    Cmodel = titanicModel(pickle.load(f))

# initialize a flask application
app = Flask("will-they-survive")

#
@app.route("/", methods=["GET"])
def hello():
    return jsonify("hello from ML API of Titanic data!")

# The API for prediction
@app.route("/will-they-survive", methods=["GET"])
def predictions():
    #load the model
    clf_path = '../models/TitanicClassifier.pkl'
    with open(clf_path, 'rb') as f:
        Cmodel = titanicModel(pickle.load(f))
        #load json
    data = request.get_json()
    if len(data.keys())==1:
        df = pd.DataFrame(data["data"])
    else:
        df = pd.read_json(json.dumps(data))
        # select the columns that are necessary for the model
    data_all_x_cols = cols
    try:
        preprocessed_df=Cmodel.prepare(df)
    except:
        return jsonify("Error occured while preprocessing your data for our model!")
    try:
        predictions= Cmodel.predict(preprocessed_df[data_all_x_cols])
    except:
        return jsonify("Error occured while processing your data into our model!")
    print("done")
    response={'data':[],'prediction_label':{1: 'survived',0: 'not survived'}}
    response['data']=int(predictions)
    # with open("../logs/log.log",'w') as f:
    #     f.write(json.dumps(response['prediction_label'][response["data"]]))
    return make_response(jsonify(response['prediction_label'][response["data"]]),200)

if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8080)