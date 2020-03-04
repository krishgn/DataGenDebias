"""This script takes in a preprocessed dataset and trains a classifier,
it finds the accuracy score and bias factor
"""
import pandas as pd

from data_operations import Data

file_name = '../data/processed_adult.csv'

data_input = Data(file_name)

X_train, X_test, y_train, y_test, Z_train, Z_test = data_input.data_split()

# initialise NeuralNet Classifier
clf_model = data_input.nn_classifier(X_train.shape[1])

# predict on test set
y_pred, roc, accuracy = data_input.predict(clf_model,X_train,X_test,y_train.values,y_test)
print(f"ROC AUC: {roc:.2f}")
print(f"Accuracy: {accuracy:.1f}%")
p_val = data_input.p_rule(y_pred.round(), Z_test)
print("The classifier satisfies the following p-rule: " + str(p_val))
