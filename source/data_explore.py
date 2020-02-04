import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from data_operations import sv_accuracy
from data_operations import create_test_train
from sklearn import preprocessing
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

sns.set(style='white', context='notebook', palette='deep')
sns.set()

#Just to get some basic info
data_orig = pd.read_csv('../data/adult.csv',low_memory=False)
print("\n \n The original columns of the input file are: \n")
print(data_orig.info())
print("\n \n The original data set looks like this: \n")
print(data_orig.head())
print("\n \n The size of the data set is \n")
print(data_orig.shape)

#Identify the numerical and categorical features
num_feat = ['age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week','income']

# Identify Categorical features
cat_feat = ['workclass','education','marital-status', 'occupation', 'relationship', 'race', 'sex', 'native']

#Fill in the missing values with the mode value of the columns
for col in ['workclass', 'occupation', 'native-country']:
    data_orig[col].fillna(data_orig[col].mode()[0], inplace=True)

# Count of >50K & <=50K
sns.countplot(data_orig['income'],label="Count")
#plt.show()

# Correlation matrix between numerical values
fig_heatmap = sns.heatmap(data_orig[num_feat].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
#plt.show()


# map the income prediction to 0 & 1
data_orig['income']=data_orig['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
#Now to compare the gender difference:
fig_gender = sns.barplot(x="gender",y="income",data=data_orig)
fig_gender = fig_gender.set_ylabel("Income >50K Probability")
#plt.show()

#Feature engineering
data_orig["gender"] = data_orig["gender"].map({"Male": 0, "Female":1})

# Create Married Column - Binary Yes(1) or No(0)
data_orig["marital-status"] = data_orig["marital-status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
data_orig["marital-status"] = data_orig["marital-status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
data_orig["marital-status"] = data_orig["marital-status"].map({"Married":1, "Single":0})
data_orig["marital-status"] = data_orig["marital-status"].astype(int)

#Create race column - binary White-1 and nonwhite-0
print(data_orig.race.unique())
data_orig["race"]= data_orig["race"].replace(['Black','Asian-Pac-Islander','Other','Amer-Indian-Eskimo'],'Minorities')
data_orig["race"]=data_orig["race"].map({"White":1,"Minorities":0})
data_orig["race"] = data_orig["race"].astype(int)

#Drop the data you don't want to use
data_orig.drop(labels=["workclass","education","occupation","relationship","native-country"], axis = 1, inplace = True)
print('Dataset with Dropped Labels')
print(data_orig.head())
print(data_orig.info())

#create two datasets, male and female and create the job ids
data_male = data_orig[(data_orig['gender'] == 1)]
data_female = data_orig[(data_orig['gender'] == 0)]

#Create test and train data based on target column TotalPay
target_name = "income"


# X_train, X_test, Y_train, Y_test = create_test_train(data_orig,data_orig)
X_train_m, X_test_m, Y_train_m, Y_test_m = create_test_train(data_male,target_name)
X_train_f, X_test_f, Y_train_f, Y_test_f = create_test_train(data_female,target_name)

acc_svc_mr_ms, pred = sv_accuracy(X_train_m,Y_train_m,X_test_m,Y_test_m)
print("\n Accuracy with male train and male test data: ",round(acc_svc_mr_ms,2),"%")


acc_svc_mr_fs, pred = sv_accuracy(X_train_m,Y_train_m,X_test_f,Y_test_f)
print("\n Accuracy with male train and female test data: ",round(acc_svc_mr_fs,2),"%")

acc_svc_fr_fs, pred = sv_accuracy(X_train_f,Y_train_f,X_test_f,Y_test_f)
print("\n Accuracy with female train and female test data: ",round(acc_svc_fr_fs,2),"%")

acc_svc_fr_ms, pred = sv_accuracy(X_train_f,Y_train_f,X_test_m,Y_test_m)
print("\n Accuracy with female train and male test data: ",round(acc_svc_fr_ms,2),"%")
