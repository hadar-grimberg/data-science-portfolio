import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer


data_x, data_y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y)

logistic = LogisticRegression(solver='saga', max_iter=1000)

penalties = ['l1', 'l2']
C_vals = [0.01, 0.1, 1]

options = []
best_score = 0
for penalty in penalties:
    for c in C_vals:
        logistic = LogisticRegression(penalty=penalty, C=c, solver='saga', max_iter=5000)
        scores = cross_val_score(logistic, x_train, y_train, cv=5)
        total_score = scores.mean()
        if total_score>best_score:
            best_score=total_score
            best_options = (c, penalty)
print('best options:', best_options)

param_grid = {'C': C_vals, 'penalty': penalties}
grid = GridSearchCV(LogisticRegression(solver='saga', max_iter=5000), param_grid)
grid.fit(x_train, y_train)
print('best params: ', grid.best_params_)

cv = KFold()

best_score = 0
for penalty in penalties:
    for c in C_vals:
        scores = []
        for train_indices, validation_indices in cv.split(x_train, y_train):
            x_train_part, y_train_part = x_train[train_indices, :], y_train[train_indices]
            x_validation_part, y_validation_part = x_train[validation_indices, :], y_train[validation_indices]
            logistic = LogisticRegression(penalty=penalty, C=c, solver='saga', max_iter=5000)
            logistic.fit(x_train_part, y_train_part)
            score = logistic.score(x_validation_part, y_validation_part)
            scores.append(score)
        total_score = np.mean(scores)
        if total_score > best_score:
            best_score = total_score
            best_options = (c, penalty)

print('best options: ', best_options)


