"""
This script asks for an augmentation factor from user
and augments the original data with TGAN generated data
Augmentation Factor = 1 is when no of males = no of females
"""
import pandas as pd
from sklearn.utils import shuffle

def number_of_synthetic_data(aug_fact, data):
    """
    This function finds the number of data points to be extracted
    from synthetic data that needs to be augmented to original data
    """
    nfemales, nmales = data.groupby('sex').size()
    ndiff = nmales/nfemales - 1 #factor of difference in male and female nos
    return round(ndiff * aug_fact * nfemales)

#input augmentation factor
aug_fact = 0.25
data_orig = pd.read_csv('../data/processed_adult.csv',low_memory=False)
datagen_female  = shuffle(pd.read_csv('../data/generated_female_15k.csv',low_memory=False))

augmentation_num = number_of_synthetic_data(aug_fact, data_orig)
#from the aug_fact add the requisite number of data points from synthetic data to the original data
data_augmented  = shuffle(data_orig.append(datagen_female[:augmentation_num], sort = False))

#write the augmented data into a new file for analysis
new_file_name = "../data/augmented_data_" + str(aug_fact) + ".csv"
data_augmented.to_csv(new_file_name,index=False)
