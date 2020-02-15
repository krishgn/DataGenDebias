import pandas as pd
import numpy as np

class Data:
    def __init__(self, data_input,  \
                protected_feature = 'sex', \
                target_feature = 'income'):
        self.data_input = data_input
        self.protected_feature = protected_feature
        self.self.target_feature = target_feature
