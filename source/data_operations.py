import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor



def create_test_train(df,target_name):
#separate target label and features and train your svm
    feat = df.drop(columns=target_name,axis=1)
    label = df[target_name]
    return train_test_split(feat, label, test_size=0.3)

def svm_create_accuracy(X_train,Y_train,X_test,Y_test):
    clf = RandomForestRegressor()
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    accuracy_train = clf.score(X_train, Y_train)
    accuracy_test = clf.score(X_test, Y_test)
    return accuracy_test
