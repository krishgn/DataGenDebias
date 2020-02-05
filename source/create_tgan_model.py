#This script
import pandas as pd
import numpy as np
from tgan.model import TGANModel

data = pd.read_csv('../data/processed_female.csv',low_memory=False)
print(data.info())
continuous_columns = [0, 2, 3 ,9, 10, 11]

tgan = TGANModel(continuous_columns)
tgan.fit(data)

model_path = '../demos/Model1.pkl'
tgan.save(model_path)
