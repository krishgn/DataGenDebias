import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from data_operations import svm_create_accuracy
from data_operations import create_test_train
from data_preprocess import create_jobid
from sklearn import preprocessing
from sklearn import utils
sns.set()

#Just to get some basic info
data_orig = pd.read_csv('../data/SF_salary_data_gender.csv',low_memory=False)
print("\n \n The original columns of the input file are: \n")
print(data_orig.info())
print("\n \n The original data set looks like this: \n")
print(data_orig.head())
print("\n \n The size of the data set is \n")
print(data_orig.shape)

#Let's check the year wise salary distribution
data_orig[["Year","TotalPay"]].groupby("Year").mean()
data_orig['Year'].plot(kind='hist')
plt.show()

#That's a lot of data and a lot of data per year. So let's just work on the 2014 salaries
data_14 = data_orig[(data_orig['Year'] == 2014)]
data_14.drop(labels=["Year"], axis = 1, inplace = True)
print("\n \n The new columns of the input file are: \n")
print(data_14.info())
print("\n \n The size of the new data set is \n")
print(data_14.shape)

#Now to compare the gender difference:
fig_gender = sns.barplot(x="gender",y="TotalPay",data=data_14)
fig_gender = fig_gender.set_ylabel("Income")
plt.show()

#Cleaning the data to keep just the essential features
data_input = data_14.filter(['JobTitle','TotalPay','gender'])
print('Input Data Labels')
print(data_input.head())

#Taking a look at the job_titles
data_input["JobTitle"].value_counts().head(10)
#Now to convert the jobtitle strings to numbers and drop the JobTitle column
JobTitle_enc = preprocessing.LabelEncoder()
JobTitle_enc.fit(data_input["JobTitle"] )
JobTitle_Lowercase = JobTitle_enc.transform(data_input["JobTitle"] )
print(type(JobTitle_Lowercase))
data_input.insert(0, "JobId", JobTitle_Lowercase.astype(int), True)
data_input.drop(labels=["JobTitle"], axis = 1, inplace = True)
print(data_input.head())

#Mapping the gender info to 0&1
data_input["gender"] = data_input["gender"].map({"m": 1, "f":0})

#create two datasets, male and female and create the job ids
data_male = data_input[(data_input['gender'] == 1)]
data_female = data_input[(data_input['gender'] == 0)]

#Create test and train data based on target column TotalPay
target_name = "TotalPay"

X_train_m, X_test_m, Y_train_m, Y_test_m = create_test_train(data_male,target_name)
X_train_f, X_test_f, Y_train_f, Y_test_f = create_test_train(data_female,target_name)

acc_svc_mr_ms = svm_create_accuracy(X_train_m,Y_train_m,X_test_m,Y_test_m)
print("\n Accuracy with male train and male test data: ",round(acc_svc_mr_ms,2),"%")

acc_svc_mr_fs = svm_create_accuracy(X_train_m,Y_train_m,X_test_f,Y_test_f)
print("\n Accuracy with male train and female test data: ",round(acc_svc_mr_fs,2),"%")

acc_svc_fr_fs = svm_create_accuracy(X_train_f,Y_train_f,X_test_f,Y_test_f)
print("\n Accuracy with female train and female test data: ",round(acc_svc_fr_fs,2),"%")

acc_svc_fr_ms = svm_create_accuracy(X_train_f,Y_train_f,X_test_m,Y_test_m)
print("\n Accuracy with female train and male test data: ",round(acc_svc_fr_ms,2),"%")
