#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# import:
import pprint
import sys
import pickle
import pandas as pd
import numpy as np
from time import time
from tester import dump_classifier_and_data

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2

import pylab as pl
from feature_format import featureFormat
from feature_format import targetFeatureSplit

import matplotlib.pyplot as plt

### Load the dictionary containing the dataset
sys.path.append("../tools/")
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
# Who is in this data set?
pprint.pprint (data_dict.keys())
print ("There are ", len(data_dict.keys()), " executives in Enron Dataset.")

    
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'bonus',
                 'salary',
                 'salary_to_avg',
                 'deferral_payments',
                 'total_payments',
                 'restricted_stock_deferred',
                 'exercised_stock_options',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
df = pd.DataFrame.from_dict(my_dataset, orient = 'index')
df = df[features_list]
df = df.replace('NaN', np.nan)
df.info()

pprint.pprint (data_dict["TOTAL"], width=1)
pprint.pprint (data_dict["THE TRAVEL AGENCY IN THE PARK"])

data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")



### Task 3: Create new feature(s)
total = 0.0
avg=0.0
for k,v in data_dict.iteritems():
    if v['salary'] == "NaN" or v['salary']<0:
        v['salary_to_avg'] = "NaN"
    else:
        total += v['salary']
avg= total/94        
print avg

for key, value in data_dict.iteritems():
    if value["salary"] == "NaN" :
        value['salary_to_avg'] = "NaN"
    else:
        value['salary_to_avg'] = float(value['salary']) / float(avg)
        
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf_GNB = GaussianNB()
model_GNB = clf_GNB.fit(features_train, labels_train) 
pred_GNB = model_GNB.predict(features_test)
acc_GNB= accuracy_score(pred_GNB, labels_test)
recall_GNB = metrics.recall_score(pred_GNB,labels_test)
precision_GNB = metrics.precision_score(pred_GNB,labels_test)

print acc_GNB, recall_GNB, precision_GNB


clf_knc = KNeighborsClassifier()
model_knc = clf_knc.fit(features_train, labels_train) 
pred_knc = model_knc.predict(features_test)
acc_knc= accuracy_score(pred_knc, labels_test)
recall = metrics.recall_score(pred_knc,labels_test)
precision = metrics.precision_score(pred_knc,labels_test)

print acc_knc, recall, precision


clf_dt = DecisionTreeClassifier(splitter='best', random_state=42)
clf_dt = clf_dt.fit(features_train, labels_train) 
pred_dt = clf_dt.predict(features_test)
acc_dt= accuracy_score(pred_dt, labels_test)
recall_dt = metrics.recall_score(pred_dt,labels_test)
precision_dt = metrics.precision_score(pred_dt,labels_test)

print acc_dt, recall_dt, precision_dt

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#Creat the pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

sss = StratifiedShuffleSplit(n_splits=100, random_state = 42)
pipe_KNC = Pipeline([
    ('reduce_dim', SelectKBest()),
    ('scaler', MinMaxScaler()),
    ('classify', KNeighborsClassifier())
])
param_grid_KNC = [
    {
        'reduce_dim__k': [3, 6, 9],
        'classify__algorithm' : ['auto'], 
        'classify__leaf_size' : [30, 50],
        'classify__n_neighbors' : [5, 10], 
        'classify__p' : [1, 2], 
        'classify__weights' : ['uniform', 'distance'],
        'classify__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
]

#gridsearch
grid_KNC = GridSearchCV(pipe_KNC, cv=sss, param_grid=param_grid_KNC, scoring='recall')
gs_KNC = grid_KNC.fit(features, labels)
clf_KNC =gs_KNC.best_estimator_
print clf_KNC




#Creat the pipeline
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=100, random_state = 42)

pipe = Pipeline([
    ('reduce_dim', SelectKBest()),
    ('scaler', MinMaxScaler()),
    ('classify', DecisionTreeClassifier(splitter='best', random_state=42))
])

param_grid_DT = [
    {
        'reduce_dim__k': [3,6,9],
        'classify__criterion':['gini', 'entropy'],
        'classify__class_weight':[None, 'balanced'],
        'classify__min_samples_split' : [2,3,4,5],
        'classify__max_leaf_nodes': [None, 5],
        'classify__min_samples_leaf': [4, 2]
    },
]

gs_DT = GridSearchCV(pipe, cv=sss, param_grid=param_grid_DT, scoring='f1')
gs_DT.fit(features, labels)
clf_DT =gs_DT.best_estimator_



finalFeatureIndices_DT = gs_DT.best_estimator_.named_steps["reduce_dim"].get_support(indices = True)
finalFeatureList_DT = [features_list[i+1] for i in finalFeatureIndices_DT]
print finalFeatureList_DT

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_DT, my_dataset, features_list)