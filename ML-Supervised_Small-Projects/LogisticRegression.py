#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: HadarG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
data1 = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                     columns=iris['feature_names'] + ['target'])
x = data1.iloc[:, :2]

blankIndex=[''] * len(x)
x.index=blankIndex


y = (pd.DataFrame([1 if x == 0 else 0 for x in iris["target"]]))


data1['Lables'] = y
plt.scatter(x.iloc[:,0], x.iloc[:,1], c=y.iloc[:,0],cmap='rainbow')
plt.show()


def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict_logit(x, theta):
    return sigmoid(np.dot(x,theta.T))>0.5

# x_values = np.linspace(-5, 5, 100)
# plt.plot(x_values, sigmoid(x_values), '-m')
# plt.show()

def log_likelihood(y,h):    #loss
    m=len(y)
    y=y.reshape(1,len(y))
    return -((y.dot(np.log(h))) + ((1-y).dot(np.log(1-h))))/m

def gradient(x,y,h):
    m=len(y)
    y = y.reshape(1, len(y))
    return (np.transpose(h - y.reshape(m, 1)).dot(x)) / m


def train_test_split (percent,x,y):
    m=len(x)
    chosen_idx_test = np.array([])
    chosen_idx_train = np.random.choice(m, replace=False, size=(int(percent*m)))
    for i in range(m):
        if i in chosen_idx_train:
            continue
        else:
            chosen_idx_test=np.append(chosen_idx_test,i)
    train_set = x.iloc[chosen_idx_train]
    test_set = x.iloc[chosen_idx_test]
    y_train = y.iloc[chosen_idx_train]
    y_test = y.iloc[chosen_idx_test]
    return train_set, test_set, y_train, y_test

train_set, test_set, y_train, y_test= train_test_split (0.8,x,y)

# train_test_split (0.8,x,y)


def logistic_regression(x,y,LR,epochs):
    x=np.array(x)
    y=np.array(y)
    theta=(np.random.normal(0,0.1,2)).reshape(1,2)
    for epoch in range(epochs):
        z = x@theta.T
        h = sigmoid(z)
        loss=log_likelihood(y,h)
        grad=gradient(x,y,h)
        theta=theta-(LR*grad)
        print('epoch {}, loss {}, new t {}'.format(epoch, loss, theta))
    return theta


theta= logistic_regression(train_set,y_train,0.1,100)

train_precision=(predict_logit(train_set, theta)==y_train).sum()/len(y_train)
test_precision=(predict_logit(test_set, theta)==y_test).sum()/len(y_test)
print('Train precision: {} Test precision: {}'.format(train_precision, test_precision))
print(theta)
