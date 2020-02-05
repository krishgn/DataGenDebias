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

from data_operations_2 import nn_classifier
from data_operations_2 import p_rule


#to differentiate numerical/categorical features for scaling/encoding
num_features = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']
cat_features = ['workclass','marital.status','occupation','relationship','race','native.country']

ct = ColumnTransformer([("scaling", StandardScaler(), num_features),\
     ("onehot", OneHotEncoder(sparse=False),cat_features)])

data_transformed = ct.fit_transform(data_orig)

print(type(data_transformed))

#
# #  sensitive attributes; we identify 'sex' as sensitive attributes
# Z = data_transformed[:, 12]
# y = data_transformed[:, 14]
# X = data_transformed[:,[0,1,2,3,4,5,6,7,8,9,10,11,13]]
#
#
# # split into train/test set
# X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, \
#                                         test_size=0.5,stratify=y, random_state=7)
#
# # initialise NeuralNet Classifier
# clf = nn_classifier(n_features=X_train.shape[1])
#
# # train on train set
# history = clf.fit(X_train, y_train, epochs=20, verbose=0)
#
# # predict on test set
# y_pred = pd.Series(clf.predict(X_test).ravel()) #, index=y_test.index)
# print(f"ROC AUC: {roc_auc_score(y_test.astype(int), y_pred):.2f}")
# accuracy = 100*sum(y_pred==y_test)/len(y_test)
# print("Accuracy: " + str(accuracy))
# #print(f"Accuracy: {100*accuracy_score(y_test, (y_pred>0.5)):.1f}%")
#
# print("The classifier satisfies the following %p-rule: " + str(p_rule(y_pred, Z_test)))
