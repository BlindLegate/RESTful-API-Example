#Author: Taha Çinkılıç

import pandas as pd
import json
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

def createModel():
    trainData = pd.read_csv("train.csv")
    trainData.drop(labels = ["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis = 1, inplace = True)

    ageImp = SimpleImputer().fit(trainData[["Age"]])
    trainData["Age"] = ageImp.transform(trainData[["Age"]])

    embImp = SimpleImputer(strategy = "most_frequent").fit(trainData[["Embarked"]])
    trainData["Embarked"] = embImp.transform(trainData[["Embarked"]])

    embEnc = OrdinalEncoder().fit(trainData[["Embarked"]])
    trainData["Embarked"] = embEnc.transform(trainData[["Embarked"]])

    sexEnc = OrdinalEncoder().fit(trainData[["Sex"]])
    trainData["Sex"] = sexEnc.transform(trainData[["Sex"]])

    ageScaler = StandardScaler().fit(trainData[["Age"]])
    trainData["Age"] = ageScaler.transform(trainData[["Age"]])

    y = trainData.pop("Survived")
    X = trainData[trainData.columns]

    parameters = [{"kernel":["linear", "rbf", "sigmoid"], "C":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
    rkfold = RepeatedKFold(n_splits = 10, n_repeats = 1, random_state = 12883823)
    clf = GridSearchCV(estimator = svm.SVC(), param_grid = parameters, scoring = "accuracy", refit = True, cv = rkfold)
    fitModel = clf.fit(X, y)
    return fitModel, embEnc, sexEnc, ageScaler

def predRes(fitModel, embEnc, sexEnc, ageScaler, values):
    framed = pd.DataFrame(data = values, index = [1])
    framed["Embarked"] = embEnc.transform(framed[["Embarked"]])
    framed["Sex"] = sexEnc.transform(framed[["Sex"]])
    framed["Age"] = ageScaler.transform(framed[["Age"]])
    pred = fitModel.predict(framed.to_dict("split")["data"]).tolist()[0]
    result = {"Prediction" : pred}
    return json.dumps(result)