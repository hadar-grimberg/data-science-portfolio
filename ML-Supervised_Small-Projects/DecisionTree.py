# -*- coding: utf-8 -*-
"""
Created on Jan 6 2020

@author: Hadar
"""

import pandas as pd
import numpy as np
from scipy.stats import stats
from sklearn.model_selection import train_test_split
import statistics
import math
import os

path = ''

file_name = 'wdbc.data'
data = pd.read_csv(file_name, header=None).values
feature_names = np.array(['radius','texture','perimeter','area','smoothness','compactness','concavity','concave points','symmetry','fractal dimension','y'])
dataset=np.hstack((data[:,2:],data[:,2].reshape(len(data),1)))
dataset[:,-1] = np.array([1 if i=='M' else 0 for i in data[:,1]])
train_set,test_set = train_test_split(dataset,train_size=0.8) # it splits the data 75%-25% by default. it is random, but if I want same data for each run train_test_split(x,y,random_state=5) mainly when I test the algorithm but not the data



def gini(dataset):
    gini_index = np.array([])
    for col in range(dataset.shape[1]):
        if col !=(dataset.shape[1]-1):
    #       M_1=np.append(M_1,statistics.mean(x[:,col]))
          mean_i=statistics.mean(dataset[:,col])
          group1=([])
          group2=([])
          for i in range(len(dataset[:,col])):
                if dataset[i,col] < mean_i:
                  group1=np.append(group1,dataset[i,-1])
                else:
                  group2=np.append(group2,dataset[i,-1])
          if len(group1)==0 or len(group2)==0:
              continue
          gini_e=((len(group1)/len(dataset))*(1-((np.count_nonzero(group1==0)/len(group1))**2+(np.count_nonzero(group1)/len(group1))**2)))+((len(group2)/len(dataset))*(1-((np.count_nonzero(group2==0)/len(group2))**2+(np.count_nonzero(group2)/len(group2))**2)))
        #     proportion = sum(i == 'M' for i in y) / len(y)
        #     gini_index = (1.0 - sum(proportion * proportion)) * (group_size / total_samples)
          gini_index=np.append(gini_index,gini_e)
    gini_val=gini_index.min()
    split_col=np.argmin(gini_index)
    split_point = statistics.mean(dataset[:, np.argmin(gini_index)])

    return split_col, split_point, gini_val



def datast_split(dataset, split_col, split_point):
    right=np.array([])
    left=np.array([])

    for row in range(dataset.shape[0]):
        if dataset[row,split_col] < split_point:
            if right.shape == (0,):
                right = np.append(right, dataset[row,:])
            else:
                right = np.vstack([right,dataset[row,:]])
        else:
            if left.shape == (0,):
                left = np.append(left, dataset[row,:])
            else:
                left = np.vstack([left, dataset[row,:]])
    return right,left
    # return {'groups':[left, left], 'feature': split_col, 'split_threshold': split_point}
# Node()

def result(leaf):
    return stats.mode(leaf)[0][0]


max_depth = 6

class Node:

    '''a node class
    Attributes:
        right_node: values under the split point
        left_node: values equal or above the split point
        split_col: best column to split by
        split_point: mean of values within the split column
        depth: depth level of the tree
        dataset:
        leaf_result:
    '''

    def __init__(self,dataset,depth):
        self.right=None
        self.left=None
        self.split_col=None
        self.split_point=None
        self.depth=depth
        self.leaf_result=None
        self.dataset=dataset
        self.gini_val=None

    def build_tree(self):
        if self.depth<=max_depth:
            split_col, split_point, gini_val=gini(self.dataset)
            self.split_col = split_col
            self.split_point = split_point
            self.gini_val = gini_val
            right_dataset, left_dataset = datast_split(self.dataset, self.split_col, self.split_point)
            if (len(right_dataset) or len (left_dataset))>=5 and self.gini_val >0.1:
                self.right = Node (right_dataset,self.depth+1)
                self.right.build_tree()
                self.left = Node(left_dataset,self.depth+1)
                self.left.build_tree()
            else:
                self.leaf_result = result(self.dataset[:,-1])
        else:
            self.leaf_result=result(self.dataset[:,-1])

    def print_tree(self):
        print('Depth: ',self.depth, self.split_col, ' ', self.leaf_result)
        if self.right:
            print('Right:')
            self.right.print_tree()
        if self.left:
            print('Left:')
            self.left.print_tree()

    def predict(self,row):
       if self.leaf_result != None:
           return self.leaf_result
       else:
           if row[self.split_col] < self.split_point:
               return self.right.predict(row)
           else:
               return self.left.predict(row)


root_node = Node(train_set,1)
root_node.build_tree()
# root_node.print_tree()


test_labels = []
x_test=test_set[:,:-1]
y_test=test_set[:,-1]
for row_data in x_test:
    row_label = root_node.predict(row_data)
    test_labels.append(row_label)

accuracy = 1-sum(abs(test_labels - y_test))/len(y_test)
