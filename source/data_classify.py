"""This script takes in a preprocessed dataset and trains a classifier,
it finds the accuracy score and bias factor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder

from data_operations import nn_classifier
from data_operations import p_rule

data_input = pd.read_csv('../data/processed_adult.csv',low_memory=False)

print(data_input.info())
print(data_input.head())

#  we identify  Z-'sex' as sensitive attributes
Z = data_input['sex']
y = data_input['income']
X = data_input
X.drop(labels=['sex','income'], axis = 1, inplace = True)

#to differentiate numerical/categorical features for scaling/encoding and
#standardize the X data
num_features = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']
cat_features = ['workclass','marital.status','occupation','relationship','race','native.country']

ct = ColumnTransformer([("scaling", StandardScaler(), num_features),\
     ("onehot", OneHotEncoder(sparse=False),cat_features)])

X = ct.fit_transform(X)

#split into train/test set
X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, \
                                         test_size=0.5,stratify=y)

# initialise NeuralNet Classifier
clf = nn_classifier(n_features=X_train.shape[1])

# train on train set
history = clf.fit(X_train, y_train.values, epochs=20, verbose=0)

# predict on test set
y_pred = pd.Series(clf.predict(X_test).ravel(), index=y_test.index)

print(f"ROC AUC: {roc_auc_score(y_test.astype(int), y_pred):.2f}")

print(f"Accuracy: {100*accuracy_score(y_test, y_pred.round()):.1f}%")

print("The classifier satisfies the following %p-rule: " + str(p_rule(y_pred.round(), Z_test)))
