import keras as ke
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import numpy as np
import pandas as pd

def nn_classifier(n_features):
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

# def p_rule(y_pred, z_values):
#     y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
#     y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
#     odds = y_z_1.mean() / y_z_0.mean()
#     #val = np.min([odds, 1/odds]) * 100
#     return odds*100

def p_rule(y_pred,z_values):
    yz_dat = pd.DataFrame(columns = ['y', 'Z'])
    yz_dat['y'] = y_pred
    yz_dat['Z'] = z_values
    num = len(yz_dat[(yz_dat['y']==1) & (yz_dat['Z']=='Male') ])
    den = len(yz_dat[(yz_dat['y']==1) & (yz_dat['Z']=='Female') ])
    return num/den
