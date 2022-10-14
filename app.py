#Author: Taha Çinkılıç

import titanic_prediction as tp
from joblib import dump, load
from flask import Flask, request
from flask_restful import Resource, Api
from requests import get, post

app = Flask(__name__)
api = Api(app)

values = {}

class takeDict(Resource):
    def get(self, info):
        print(type(values))
        return values[info]
    def post(self, info):
        values[info] = request.json
        return values

class train(Resource):
    def get(self, info):
        dump(tp.createModel(), f"{info}.model")
        return "Successfully trained."

class predict(Resource):
    def get(self, info):
        fitModel, embEnc, sexEnc, ageScaler = load(f"{info}.model")
        result = tp.predRes(fitModel, embEnc, sexEnc, ageScaler, values[info])
        return result

api.add_resource(takeDict, "/<string:info>")
api.add_resource(train, "/train/<string:info>")
api.add_resource(predict, "/train/predict/<string:info>")

if __name__ == '__main__':
    app.run(debug=True)

"""
---Console Command Samples---
python
import json
from requests import post, get
post("http://127.0.0.1:5000/Taha", json = {"Pclass": "2", "Sex": "male", "Age": 23.0, "SibSp": "2", "Parch": "0", "Embarked": "S"}).json()
get("http://127.0.0.1:5000/Taha").json()
get("http://127.0.0.1:5000/train/Taha").json()
get("http://127.0.0.1:5000/train/predict/Taha").json()
"""