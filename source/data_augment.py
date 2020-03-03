"""
This script asks for an augmentation factor from user
and augments the original data with TGAN generated data
Augmentation Factor = 1 is when no of males = no of females
"""
import pandas as pd
from sklearn.utils import shuffle


aug_fact = 0.25

data_orig = pd.read_csv('../data/processed_adult.csv',low_memory=False)
datagen_female  = pd.read_csv('../data/generated_female_15k.csv',low_memory=False)

nfemales, nmales = data_orig.groupby('sex').size()

ndiff = nmales/nfemales - 1

ndata = ndiff * aug_fact

datagen_female = shuffle(datagen_female)
data_new = datagen_female[:round(ndata)]

data_augmented  = data_orig.append(data_new,sort = False)

data_augmented = shuffle(data_orig)
