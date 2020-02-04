import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def create_test_train(df,target_name):
"""This function separates target label and features
    to be trained by svm"""
    feat = df.drop(columns=target_name,axis=1)
    label = df[target_name]
    return train_test_split(feat, label, test_size=0.3)

def sv_accuracy(X_train,Y_train,X_test,Y_test):
"""This functions takes train and test data as input.
    An SV classifier is trained and used to predict on test data
    Accuracy calculated on prediction and Y_test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform( X_train )
    X_test_scaled = scaler.transform( X_test )
    svc = SVC(kernel = 'rbf', max_iter = 1000, probability = True)
    svmodel = svc.fit(X_train_scaled, Y_train)
    pred = svmodel.predict(X_test_scaled)
    accuracy_test = accuracy_score(Y_test, pred)
    return accuracy_test, pred
