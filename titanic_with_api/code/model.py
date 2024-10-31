# -*- coding: utf-8 -*-
"""
Hadar Grimberg
10/2/2021

"""

import pandas as pd
from model_selection import import_data, model_select, hyperparameters_tuning, retrain
from preprocessing import data_prepare
import optuna
import pickle

class titanicModel(object):

    def __init__(self, model):
        """Simple NLP
        Attributes:
            clf: the ensemble model

        """
        self.clf = model

    def train(self, x_train, y_train):
        """Trains the classifier to associate the label
        """
        self.clf.fit(x_train, y_train)

    def prepare(self,new_data):
        train = pd.read_csv(r"..\data\raw\train.csv", index_col=0)
        test = pd.read_csv(r"..\data\raw\test.csv", index_col=0)
        full = pd.concat([train, test, new_data], axis=0)
        full=data_prepare(full)
        # full.drop("Survived", axis=1, inplace=True)
        return full.iloc[[-1]]

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def accuracy(self, X, y):
        """Returns the accuracy of predictions
        """
        score = self.clf.score(X, y)
        return score

    def pickle_clf(self, path='../models/TitanicClassifier.pkl'):
            """Saves the trained classifier for future use.
            """
            with open(path, 'wb') as f:
                pickle.dump(self.clf, f)
                print("Pickled classifier at {}".format(path))


if __name__ == '__main__':
    pass