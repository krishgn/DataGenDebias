"""This script cleans the raw data "adult.csv" and does some exploratory survey.
It creates 3 output csv files.
  i) Clean full dataset
  ii) Clean male only dataset
  iii) Clean female only dataset"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_orig = pd.read_csv('../data/adult.csv',low_memory=False)

#Just to get some basic info
data_orig = pd.read_csv('../data/adult.csv',low_memory=False)
print("\n \n The original columns of the input file are: \n")
print(data_orig.info())
print("\n \n The original data set looks like this: \n")
print(data_orig.head())
print("\n \n The size of the data set is \n")
print(data_orig.shape)

sns.countplot(data_orig['sex'],label="Count")
plt.show()

# Dropping the duplicate Rows
data_orig = data_orig.drop_duplicates(keep = 'first')
#removing the education column because education.num is numerical value of the same
data_orig.drop(labels="education", axis = 1, inplace = True)

#Removing rows that has missing values in occupation, workclass or native.country column
data_orig = data_orig.drop(data_orig[data_orig['occupation']=='?'].index)
data_orig = data_orig.drop(data_orig[data_orig['workclass']=='?'].index)
data_input = data_orig.drop(data_orig[data_orig['native.country']=='?'].index)
print(data_input.shape)

# Count of >50K & <=50K
sns.countplot(data_input['income'],label="Count")
plt.show()

# map the income prediction to 0 & 1
data_input['income']=data_input['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1}).astype(int)

#create two datasets, male and female
data_male = data_input[(data_orig['sex'] == 'Male')]
data_female = data_input[(data_orig['sex'] == 'Female')]

#write all the outputs data to csv files
data_input.to_csv('../data/processed_adult.csv',header=False, index=False)
data_male.to_csv('../data/processed_male.csv', header=False, index=False)
data_female.to_csv('../data/processed_female.csv', header=False, index=False)
