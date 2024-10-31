# -*- coding: utf-8 -*-
"""
Hadar Grimberg
10/2/2021

"""

import pandas as pd
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna


def import_data():
    x_train = pd.read_excel(r"..\data\processed\x_train.xlsx", index_col=0)
    x_val = pd.read_excel(r"..\data\processed\x_val.xlsx", index_col=0)
    y_train = pd.read_excel(r"..\data\processed\y_train.xlsx", index_col=0)
    y_val = pd.read_excel(r"..\data\processed\y_val.xlsx", index_col=0)
    test = pd.read_excel(r"..\data\processed\test.xlsx", index_col=0)
    # drop similar coloumns ('SibSp', 'Parch', 'FamilyS' engineered to 'FamilyS_g_Single',
    # 'FamilyS_g_OneFM', 'FamilyS_g_SmallF', 'FamilyS_g_MedF')
    x_train.drop(['SibSp', 'Parch', 'FamilyS'], axis=1, inplace=True)
    x_val.drop(['SibSp', 'Parch', 'FamilyS'], axis=1, inplace=True)
    return x_train, x_val, y_train, y_val, test

def model_select(x_train, y_train):
    # finding the best estimator from 14 classifiers
    random_state = 17

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "Extra Trees", "Gradient Boosting", "Logistic Regression", "LDA", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=random_state),
        SVC(gamma=2, C=1, random_state=random_state),
        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=random_state),
        DecisionTreeClassifier(max_depth=5, random_state=random_state),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1,random_state=random_state),
        MLPClassifier(alpha=1, max_iter=1000, random_state=random_state),
        AdaBoostClassifier(random_state=random_state),
        GaussianNB(),
        ExtraTreesClassifier(random_state=random_state),
        GradientBoostingClassifier(random_state=random_state),
        LogisticRegression(random_state=random_state),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    # Cross validate model with Kfold stratified cross val
    kfold = StratifiedKFold(n_splits=10)

    cv_results = []
    for classifier in classifiers:
        cv_results.append(cross_val_score(classifier, x_train, y=y_train, scoring="accuracy", cv=kfold))

    # Mean the results of each model 10-k folds
    CVmeans = []
    CVstd = []
    for cv_result in cv_results:
        CVmeans.append(cv_result.mean())
        CVstd.append(cv_result.std())
    #sort and show results
    CVtable = pd.DataFrame({ "Algorithm": names, "CrossValMeans": CVmeans, "CrossValerrors": CVstd}).sort_values(by=['CrossValMeans'],ascending=False)
    print(CVtable)
    # 6 models had accurarcy of 0.8 or above. The models will take forward for hyper-parameters tuning
    classifiersTuning = [classifiers[cls] for cls in CVtable.index[CVtable.CrossValMeans > 0.8].to_list()]
    namesTuning = [names[cls] for cls in CVtable.index[CVtable.CrossValMeans > 0.8].to_list()]

    return classifiersTuning, namesTuning

def hyperparameters_tuning(trial, x_train, y_train, namesTuning):
    # find the best model with hyperparameters  tuning using optuna
    classifier_name = trial.suggest_categorical("classifier", namesTuning)
    ## pay attention that hyperparameters are specific for the obtained results
    if classifier_name == namesTuning[0]:
        GB_est = trial.suggest_int("GB_est", 100, 300, log=True)
        GB_lr = trial.suggest_float("GB_lr", 1e-5, 1e-1, log=True)
        GB_max_depth = trial.suggest_int("GB_max_depth", 2, 10, log=True)
        GB_min_samples = trial.suggest_int("GB_min_samples", 2, 50, log=True)
        GB_max_features = trial.suggest_float("GB_max_features", 0.1, 0.5, log=True)
        classifier_obj = GradientBoostingClassifier(random_state=17, n_estimators= GB_est,
        learning_rate=GB_lr, max_depth=GB_max_depth , min_samples_leaf=GB_min_samples,
                                                    max_features= GB_max_features)
    elif classifier_name == namesTuning[1]:
        LR_max_iter = trial.suggest_int("LR_max_iter", 50, 350, log=True)
        LR_C = trial.suggest_float("LR_C", 1, 1000, log=True)
        LR_tol = trial.suggest_float("LR_tol", 1e-5, 1, log=True)
        LR_penalty = trial.suggest_categorical("LR_penalty",["l2", "none"])
        classifier_obj = LogisticRegression(random_state=17, max_iter= LR_max_iter,
        tol=LR_tol, C=LR_C, penalty= LR_penalty)
    elif classifier_name == namesTuning[2]:
        GP_max_iter = trial.suggest_int("GP_max_iter", 50, 350, log=True)
        GP_restarts = trial.suggest_int("GP_restarts", 1, 20, log=True)
        classifier_obj = GaussianProcessClassifier(random_state=17,
        n_restarts_optimizer=GP_restarts, max_iter_predict=GP_max_iter)
    elif classifier_name == namesTuning[3]:
        LDA_solver = trial.suggest_categorical("LDA_solver", ['svd', 'lsqr'])
        LDA_tol = trial.suggest_float("LDA_tol", 1e-5, 1, log=True)
        classifier_obj = LinearDiscriminantAnalysis(tol=LDA_tol , solver=LDA_solver)
    elif classifier_name == namesTuning[4]:
        AB_est = trial.suggest_int("AB_est", 1, 50, log=True)
        AB_lr = trial.suggest_float("AB_lr", 1e-4, 1.5, log=True)
        AB_algo = trial.suggest_categorical("AB_algo", ["SAMME", "SAMME.R"])
        AB_est_split = trial.suggest_categorical("AB_est_split", ["best", "random"])
        AB_est_crit = trial.suggest_categorical("AB_est_crit", ["gini", "entropy"])
        classifier_obj = AdaBoostClassifier(DecisionTreeClassifier(criterion=AB_est_crit, splitter= AB_est_split),
        random_state=17, n_estimators= AB_est, learning_rate=AB_lr, algorithm=AB_algo)
    elif classifier_name == namesTuning[5]:
        DT_min_leaf = trial.suggest_int("DT_min_leaf", 2, 50, log=True)
        DT_max_depth = trial.suggest_int("DT_max_depth", 2, 10, log=True)
        DT_split = trial.suggest_categorical("DT_split", ["best", "random"])
        DT_crit = trial.suggest_categorical("DT_crit", ["gini", "entropy"])
        classifier_obj = DecisionTreeClassifier(random_state=17, min_samples_leaf= DT_min_leaf,
        criterion=DT_crit, max_depth=DT_max_depth, splitter= DT_split)

    score = cross_val_score(classifier_obj, x_train, y_train, cv=10)
    accuracy = score.mean()
    return accuracy

def retrain(tr_best, x_train, x_val, y_train, y_val):
    Gradient_Boosting = tr_best[tr_best.params_classifier == 'Gradient Boosting'].dropna(axis=1)
    Logistic_Regression = tr_best[tr_best.params_classifier == 'Logistic Regression'].dropna(axis=1)
    Gaussian_Process = tr_best[tr_best.params_classifier == 'Gaussian Process'].dropna(axis=1)
    L_D_A = tr_best[tr_best.params_classifier == 'LDA'].dropna(axis=1)
    AdaBoost = tr_best[tr_best.params_classifier == 'AdaBoost'].dropna(axis=1)
    Decision_Tree = tr_best[tr_best.params_classifier == 'Decision Tree'].dropna(axis=1)

    # since all models got the same accurarcy, one will be chosen randomly and retrain
    rand = random.choice(Gradient_Boosting.index.tolist())
    GBC = GradientBoostingClassifier(n_estimators=int(Gradient_Boosting.loc[rand,"params_GB_est"]),
          learning_rate=Gradient_Boosting.loc[rand,"params_GB_lr"], max_depth=Gradient_Boosting.loc[rand,"params_GB_max_depth"],
          min_samples_leaf=int(Gradient_Boosting.loc[rand,"params_GB_min_samples"]), max_features=Gradient_Boosting.loc[rand,"params_GB_max_features"])
    GBC.fit(x_train,y_train)
    print(f"Gradient Boosting train accuracy: {GBC.score(x_train,y_train):0.3f}, validation accuracy {GBC.score(x_val,y_val):0.3f}")

    rand = random.choice(Logistic_Regression.index.tolist())
    LR = LogisticRegression(max_iter=Logistic_Regression.loc[rand, 'params_LR_max_iter'],
     tol=Logistic_Regression.loc[rand, 'params_LR_tol'], C=Logistic_Regression.loc[rand, "params_LR_C"], penalty=Logistic_Regression.loc[rand, 'params_LR_penalty'])
    LR.fit(x_train, y_train)
    print(f"Logistic Regression train accuracy: {LR.score(x_train, y_train):0.3f}, validation accuracy {LR.score(x_val, y_val):0.3f}")

    rand = random.choice(Gaussian_Process.index.tolist())
    GP = GaussianProcessClassifier(n_restarts_optimizer=int(Gaussian_Process.loc[rand, 'params_GP_restarts']), max_iter_predict=int(Gaussian_Process.loc[rand, 'params_GP_max_iter']))
    GP.fit(x_train, y_train)
    print(f"Gaussian Process train accuracy: {GP.score(x_train, y_train):0.3f}, validation accuracy {GP.score(x_val, y_val):0.3f}")

    rand = random.choice(L_D_A.index.tolist())
    LDA = LinearDiscriminantAnalysis(tol=L_D_A.loc[rand, 'params_LDA_tol'], solver=L_D_A.loc[rand, 'params_LDA_solver'])
    LDA.fit(x_train, y_train)
    print(f"Linear Discriminant Analysis train accuracy: {LDA.score(x_train, y_train):0.3f}, validation accuracy {LDA.score(x_val, y_val):0.3f}")

    rand = random.choice(AdaBoost.index.tolist())
    AB = AdaBoostClassifier(DecisionTreeClassifier(criterion=AdaBoost.loc[rand, 'params_AB_est_crit'], splitter=AdaBoost.loc[rand, 'params_AB_est_split']),
         n_estimators=int(AdaBoost.loc[rand, 'params_AB_est']), learning_rate=AdaBoost.loc[rand, 'params_AB_lr'], algorithm=AdaBoost.loc[rand, 'params_AB_algo'])
    AB.fit(x_train, y_train)
    print(f"AdaBoost train accuracy: {AB.score(x_train, y_train):0.3f}, validation accuracy {AB.score(x_val, y_val):0.3f}")

    rand = random.choice(Decision_Tree.index.tolist())
    DT =  DecisionTreeClassifier(min_samples_leaf=int(Decision_Tree.loc[rand, "params_DT_min_leaf"]),  splitter=Decision_Tree.loc[rand, "params_DT_split"],
          criterion=Decision_Tree.loc[rand, "params_DT_crit"], max_depth=Decision_Tree.loc[rand, "params_DT_max_depth"])
    DT.fit(x_train, y_train)
    print(f"Decision Tree train accuracy: {DT.score(x_train, y_train):0.3f}, validation accuracy {DT.score(x_val, y_val):0.3f}")

    # return the top 5 predictors:
    models = [GBC, LR,  GP, LDA, AB, DT]
    scores = np.array([GBC.score(x_val, y_val), LR.score(x_val, y_val), GP.score(x_val, y_val),
            LDA.score(x_val, y_val), AB.score(x_val, y_val), DT.score(x_val, y_val)])
    models.pop(np.argmin(scores))
    return models


if __name__ == '__main__':
    # Load the preprocessed data
    x_train, x_val, y_train, y_val, test = import_data()
    # Select the best model and hyper-parameters
    classifiersTuning, namesTuning = model_select(x_train, y_train)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: hyperparameters_tuning(trial, x_train, y_train, namesTuning) , n_trials=500)
    print(study.best_trial)
    # get the best hyper pameters for each model and retrain with the entaire dataset
    tr = study.trials_dataframe()
    idx = tr.groupby(['params_classifier'])['value'].transform(max) == tr['value']
    tr_best = tr[idx]

    models = retrain(tr_best, x_train, x_val, y_train, y_val)