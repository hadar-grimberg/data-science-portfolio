# -*- coding: utf-8 -*-
"""
Created on Jan 6 2020

@author: Hadar Grimberg
"""

import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
import os

path = ''
#os.chdir(path)

#1
file_name = r'C:\Users\Hadar Grimberg\PycharmProjects\ML\Supervised\pima-indians-diabetes.csv'
data = pd.read_csv(file_name,header=None).values
x = data[:,:-1] # all the columns but the last one ## data.iloc[] in pandas - takes the columns by index loc-by names
y = data[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y) # it splits the data 75%-25% by default. it is random, but if I want same data for each run train_test_split(x,y,random_state=5) mainly when I test the algorithm but not the data

#2
mean_x = np.mean(x_train,axis=0)
std_x = np.std(x_train,axis=0)

def statistics_per_label(x,y,label):
    x_with_specific_label = x[y == label]
    mean=np.mean(x_with_specific_label,axis=0)
    std = np.std(x_with_specific_label, axis=0)
    return mean,std

def calc_mean_std(x,y):
    mean_0, std_0 = statistics_per_label(x, y, 0)
    mean_1, std_1 = statistics_per_label(x, y, 1)
    return mean_0, std_0, mean_1, std_1

#taken from equation of probability of Gaussian
def predict_gussian(x,mean,std):
    prob_array = (1/(np.sqrt(2*math.pi)*std))*np.exp(-(x-mean)**2/(2*std**2))
    prob = np.prod(prob_array,axis=1)
    prob = np.reshape(prob, (prob.shape[0], 1))
    return prob

def predictions(x,y):
    mean_0, std_0, mean_1, std_1=calc_mean_std(x,y)
    prob_0 = predict_gussian(x,mean_0,std_0)
    prob_1 = predict_gussian(x,mean_1,std_1)
    probabilities = np.column_stack((prob_0,prob_1))
    y_calculated=np.argmax(probabilities,axis=1)
    return y_calculated

def accurarcy(y_calculated,y):
    y_0=0
    y_0_correct=0
    y_1 = 0
    y_1_correct = 0
    for i in range(len(y)):
        if y[i]==0:
            y_0+=1
            if y_calculated[i]==y[i]:
                y_0_correct+=1
        else:
            y_1+=1
            if y_calculated[i]==y[i]:
                y_1_correct+=1
    pred_0=y_0_correct/y_0
    pred_1=y_1_correct/y_1
    pred=(y_0_correct+y_1_correct)/(y_0+y_1)
    return (pred_0,pred_1,pred)


class NaiveBayesClassifier():

    def __init__(self):
        self.alg_name = 'NaiveBayes'
        self.mean0 = ''
        self.std0 = ''
        self.mean1 = ''
        self.std1 = ''

    def fit(self, x_train, y_train):
        mean0, std0, mean1, std1 = calc_mean_std(x_train, y_train)
        self.mean0 = mean0
        self.mean1 = mean1
        self.std0 = std0
        self.std1 = std1

    def predict(self, x_test, y_test):
        prob_0 = predict_gussian(x_test, self.mean0, self.std0)
        prob_1 = predict_gussian(x_test, self.mean1, self.std1)
        probabilities = np.column_stack((prob_0, prob_1))
        y_calculated = np.argmax(probabilities, axis=1)
        self.y_calculated = y_calculated
        pred_0, pred_1, pred = accurarcy(y_calculated, y_test)
        return pred_0,pred_1,pred


if __name__ == '__main__':
    my_classifier = NaiveBayesClassifier()
    my_classifier.fit(x_train, y_train)
    accuracy = my_classifier.predict(x_test, y_test)
