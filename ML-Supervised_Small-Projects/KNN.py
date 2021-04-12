#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: HadarG
"""

import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from statistics import mode

def build_dataset():
    iris = load_iris()
    data1 = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                         columns=iris['feature_names'] + ['target'])
    m = len(data1)
    chosen_idx_test = np.array([])
    chosen_idx_train = np.random.choice(m, replace=False, size=(int(0.8 * m)))
    for i in range(m):
        if i in chosen_idx_train:
            continue
        else:
            chosen_idx_test = np.append(chosen_idx_test, i)
    np.random.shuffle(chosen_idx_test)
    train_set = data1.iloc[chosen_idx_train,:-1]
    test_set = data1.iloc[chosen_idx_test,:-1]
    y_train = data1.iloc[chosen_idx_train,-1]
    y_test = data1.iloc[chosen_idx_test,-1]
    return train_set, test_set, y_train, y_test

def distance(x,y):
    return np.linalg.norm(x-y)

def Nearest_neighbours(k,x,y):
    dist=np.array([])
    f = np.array([])
    column_names=np.array([])
    for i in range(len(x)):
        column_names = np.append(column_names, x.index.values[i])
        for j in range(len(y)):
            dist=np.append(dist,distance(x.iloc[i],y.iloc[j]))
    dists = dist.reshape(len(train_set) , len(train_set))
    distances = pd.DataFrame(dists, columns=column_names, index=column_names)
    for idx in range(len(distances)):
        distances.sort_values(by=distances.columns.values[idx], inplace=True)
        m=1
        while m!=(k+1):
            f=np.append(f,distances.index.values[m])
            m+=1
    f=f.reshape(len(distances),k)
    k_nearest = pd.DataFrame(f, index=column_names)
    return distances, k_nearest

def voting(k_nearest,y):
    mode_list=np.array([])
    for i,j in k_nearest.iterrows():
        p = np.array([])
        for rr in j:
            p=np.append(p,y.xs(rr))
        mode_list=np.append(mode_list,mode(p))
    return mode_list

def prediction(mode_list,y):
    m=0
    for i in range(len(mode_list)):
        if mode_list[i]==y.iloc[i]:
            m+=1
        else:
            continue
    return m/len(y)




if __name__ == '__main__':
    train_set, test_set, y_train, y_test = build_dataset()
    distances, k_nearest = Nearest_neighbours(3, train_set, train_set)
    mode_list = voting(k_nearest, y_train)
    accurarcy = prediction(mode_list, y_train)
