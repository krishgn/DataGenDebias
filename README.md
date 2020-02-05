# DataGenDebias
To mitigate bias in dataset by generating synthetic data using TGAN

The scripts can be found in source/ and are as follows
1. data_clean.py : This script cleans the raw data and does some exploratory survey. It creates 3 output csv files.
  i) Clean full dataset
  ii) Clean male only dataset
  iii) Clean female only dataset
2. data_classify.py : This script runs the nn classifier on the data and analyses the results. Calculates accuracy, bias metric, etc.
3. create_TGAN_model.py : This script takes in the clean data and trains the TGAN on it to create a model
4. generate_TGAN_data.py : This scripts takes in the already trained TGAN model and generates the requisite number of synthetic data samples
5. data_operations.py : This script contains the different functions used in data_classify
