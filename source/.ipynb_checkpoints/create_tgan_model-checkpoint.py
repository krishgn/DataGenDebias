#This script

import pandas as pd
import numpy as np
from tgan.model import TGANModel

data = pd.read_csv('../data/census-train.csv',low_memory=False)
continuous_columns = [0, 5, 16, 17, 18, 29, 38]

tgan = TGANModel(continuous_columns)
tgan.fit(data)

model_path = '../demos/Model1'
tgan.save(model_path)
