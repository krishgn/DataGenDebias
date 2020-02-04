import pandas as pd
import numpy as np
from tgan.model import TGANModel

model_path = '../demos/Model1'

num_samples = 1000
new_tgan = TGANModel.load(model_path)
new_samples = new_tgan.sample(num_samples)

print(type(samples))
print(samples.size())
