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
import keras as ke
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model

class Data():
    def __init__(self, file_name,  \
                protected_feature = 'sex', \
                target_feature = 'income'):
        self.data_input = pd.read_csv(file_name,low_memory=False)
        self.protected_feature = protected_feature
        self.target_feature = target_feature
        self.Z = self.data_input[self.protected_feature]
        self.y = self.data_input[self.target_feature]
        self.X = self.data_input
        self.X.drop(labels=['sex','income'], axis = 1, inplace = True)

    def data_split(self):
        """
        To differentiate numerical/categorical features for scaling/encoding and
        standardize the X data. To split input data to train and test.
        """
        num_features = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']
        cat_features = ['workclass','marital.status','occupation','relationship','race','native.country']
        ct = ColumnTransformer([("scaling", StandardScaler(), num_features),\
             ("onehot", OneHotEncoder(sparse=False),cat_features)])
        self.X = ct.fit_transform(self.X)
        return train_test_split(self.X, self.y, self.Z, test_size=0.5,stratify=self.y)

    def nn_classifier(self,n_features):
        inputs = Input(shape=(n_features,))
        dense1 = Dense(32, activation='relu')(inputs)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation="relu")(dropout2)
        dropout3 = Dropout(0.2)(dense3)
        outputs = Dense(1, activation='sigmoid')(dropout3)
        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def predict(self, clf_model,X_train,X_test,y_train,y_test):
        history = clf_model.fit(X_train, y_train, epochs=20, verbose=0)
        y_pred = pd.Series(clf_model.predict(X_test).ravel(), index=y_test.index)
        return y_pred, roc_auc_score(y_test.astype(int), y_pred), \
            100*accuracy_score(y_test, y_pred.round())

    def p_rule(self,y_pred,z_values):
        yz_dat = pd.DataFrame(columns = ['y', 'Z'])
        yz_dat['y'] = y_pred
        yz_dat['Z'] = z_values
        num = len(yz_dat[(yz_dat['y']==1) & (yz_dat['Z']=='Male') ])
        den = len(yz_dat[(yz_dat['y']==1) & (yz_dat['Z']=='Female') ])
        return num/den
