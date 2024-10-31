# -*- coding: utf-8 -*-
"""
Hadar Grimberg
10/3/2021

"""

# -*- coding: utf-8 -*-
"""
Hadar Grimberg
10/2/2021

"""


from model_selection import import_data, model_select, hyperparameters_tuning, retrain
import optuna
import pickle
from sklearn.ensemble import VotingClassifier
from model import titanicModel


def ensemble(models):
    # voting classifier to combine the predictions combining from the 5 classifiers.
    classifier = VotingClassifier(estimators=[(str(models[0]).split("(")[0], models[0]),
                (str(models[1]).split("(")[0], models[1]), (str(models[2]).split("(")[0], models[2]),
                (str(models[3]).split("(")[0], models[3]), (str(models[4]).split("(")[0], models[4])], voting='soft')
    return classifier


if __name__ == '__main__':
    # Load the preprocessed data
    x_train, x_val, y_train, y_val, test = import_data()
    # Select the best model and hyper-parameters
    classifiersTuning, namesTuning = model_select(x_train, y_train)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: hyperparameters_tuning(trial, x_train, y_train, namesTuning) , n_trials=500)
    print(study.best_trial)
    # get the best hyperpameters for each model and retrain with the entire dataset
    tr = study.trials_dataframe()
    idx = tr.groupby(['params_classifier'])['value'].transform(max) == tr['value']
    tr_best = tr[idx]
    models = retrain(tr_best, x_train, x_val, y_train, y_val)
    # combine the predictions of 5 models into a voting classifier
    classifier = ensemble(models)

    # upload the model into an object and save it
    theModel = titanicModel(classifier)
    # train the final model
    theModel.train(x_train, y_train)
    # make predictions
    y_prob = theModel.predict_proba(x_val)
    y_pred = theModel.predict(x_val)
    accuracy = theModel.accuracy(x_val,y_val)

    # finally save the model
    theModel.pickle_clf()

    # Done =)