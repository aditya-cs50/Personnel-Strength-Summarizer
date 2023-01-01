#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np


# In[2]:


def log_loss_list(x):
    y = []
    for element in x:
        y.append (element**2 -2*element)
    return y

def log_loss(x):
    y = x**2 -2*x
    return y

x = np.linspace(-3,5,100)
y = log_loss(x)

plt.plot(x,y)

def gradient(x):
    return 2*x-2

x_old = 4.5
array = [x_old]
x_old_1 = 0

for i in range(10):
    x_old_1 = x_old
    x_old = x_old - gradient(x_old)*0.8
    array.append(x_old)

plt.plot(array, log_loss_list(array), "-o")


# In[3]:


from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[4]:


X_syn, Y_syn = make_classification(n_samples=1000, n_features=200, n_informative=3, n_redundant=10, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y =0.01, class_sep=0.8, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=24)
print(X_syn.shape, Y_syn.shape)


# In[5]:


for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(X_syn[:, i])


# In[6]:


#demo of overfitting (high auc for training data, low auc for testing data)
x_train, x_test, y_train, y_test = train_test_split(X_syn, Y_syn, test_size = 0.2, random_state=1)
lr = LogisticRegression(solver="liblinear", C= 1000, penalty = "l1")
lr.fit(x_train,y_train)
proba = lr.predict_proba(x_train)[:,1]
roc_auc_score(y_train, proba)


# In[7]:


from sklearn.model_selection import StratifiedKFold, KFold
k_folds = StratifiedKFold(n_splits = 4, shuffle= True, random_state=1)


C_values_exp = [i for i in range(-3,4)]
C_values = [10**i for i in C_values_exp]


iterator = k_folds.split(X_syn, Y_syn)

auc_values = np.zeros((7, 9))
c_value_counter = 0

print(auc_values)

for C_value in C_values:
    roc_counter = 1

    lr = LogisticRegression(solver="liblinear", C=C_value)
    auc_values[c_value_counter][0] = C_value

    for train_index, test_index in iterator:   #for each fold

        x_train = [X_syn[i] for i in train_index]
        y_train = [Y_syn[i] for i in train_index]

        x_test = [X_syn[i] for i in test_index]
        y_test = [Y_syn[i] for i in test_index]

        lr.fit(x_train, y_train)
                
        proba = lr.predict_proba(x_train)[:, 1]
        train_roc = roc_auc_score(y_train, proba)
        auc_values[c_value_counter][roc_counter] = train_roc
        print(roc_counter)
        roc_counter += 1

        proba = lr.predict_proba(x_test)[:, 1]
        test_roc = roc_auc_score(y_test, proba)
        auc_values[c_value_counter][roc_counter] = test_roc
        print(roc_counter)
        roc_counter += 1

        
    c_value_counter += 1
    
print(auc_values)

plt.plot(auc_values[:,0].shape, auc_values[:,0].shape)


# In[8]:


from sklearn.model_selection import StratifiedKFold, KFold
k_folds = StratifiedKFold(n_splits = 4, shuffle= True, random_state=1)


C_values_exp = [i for i in range(-3,4)]
C_values = [10**i for i in C_values_exp]

    
    #for each C-value, train a LR model and obtain ROC values for each of 4 folds, for training data.
iterator = k_folds.split(X_syn, Y_syn)

auc_values = np.empty((7, 9))
c_value_counter = 0
print (auc_values)

for C_value in C_values:
    roc_counter = 0

    lr = LogisticRegression(solver="liblinear", C=C_value)
    auc_values[0][c_value_counter] = C_value

    for train_index, test_index in iterator:   #for each fold

        x_train = [X_syn[i] for i in train_index]
        y_train = [Y_syn[i] for i in train_index]

        x_test = [X_syn[i] for i in test_index]
        y_test = [Y_syn[i] for i in test_index]

        lr.fit(x_train, y_train)

        proba = lr.predict_proba(x_train)
        train_roc = roc_auc_score(y_train, proba)
        auc_values[c_value_counter][roc_counter] = train_roc

        roc_counter += 1
        
    c_value_counter += 1

print (auc_values)

function(k_folds, C_values, X_syn, Y_syn)
            
            
            
            
    
#for each C-value, train a LR model and obtain ROC values for each of 4 folds, for training data.
    
#plot graph of ROC value against C-value (8 points per C-value - 4 folds, and each fold has training and testing ROC values)
    
    
    



    
    


# In[ ]:


from sklearn.model_selection import StratifiedKFold, KFold
k_folds = StratifiedKFold(n_splits = 4, shuffle= True, random_state=1)


C_values_exp = [i for i in range(-3,4)]
C_values = [10**i for i in C_values_exp]

    
def function(k_folds, C_values, X, Y):
    #for each C-value, train a LR model and obtain ROC values for each of 4 folds, for training data.
    for C_value in C_values:
    roc_counter = 1

    lr = LogisticRegression(solver="liblinear", C=C_value)
    auc_values[c_value_counter][0] = C_value
    print(auc_values)

    for train_index, test_index in iterator:   #for each fold

        x_train = [X_syn[i] for i in train_index]
        y_train = [Y_syn[i] for i in train_index]

        x_test = [X_syn[i] for i in test_index]
        y_test = [Y_syn[i] for i in test_index]

        lr.fit(x_train, y_train)

        proba = lr.predict_proba(x_train)[:, 1]
        train_roc = roc_auc_score(y_train, proba)
        auc_values[c_value_counter][roc_counter] = train_roc

        roc_counter += 1

        proba = lr.predict_proba(x_test)[:, 1]
        train_roc = roc_auc_score(y_test, proba)
        auc_values[c_value_counter][roc_counter] = train_roc

        roc_counter += 1


    c_value_counter += 1





    #for each C-value, train a LR model and obtain ROC values for each of 4 folds, for training data.

    #plot graph of ROC value against C-value (8 points per C-value - 4 folds, and each fold has training and testing ROC values)






    
    


# In[ ]:




