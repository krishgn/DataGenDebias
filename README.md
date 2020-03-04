# **DataGenDebias**

### 1. Overview
To mitigate bias in dataset by generating synthetic data using TGAN.

#### 1.1 Motivation

We live in a world where ML models are increasingly used for every day decision making. With this growing impact there are also growing concerns on the fairness of these models, specifically on how ML can discriminate against population minorities. Be it in professional or financial spheres.
This is primarily because the ML system that us trained to a certain task feeds on a dataset that might come with an inherent bias. So a male dominated working space continues to be a male dominated working space. However multiple studies have shown that diversity increases performance in work spaces. This diversity can only be attained by recruiting diverse teams immune to recruitment bias. Thus it becomes important to debias our datasets before it enters an ML model.

#### 1.2 Approach
In DataGenDebias we make use of Synthetic Data Generation by Tabular GAN (citation). TGANs are optimized to learn the correlation between different columns in a tabular data and generate synthetic samples that mimick the original source. This has shown to reduce the bias by more than 200% with a trade-off in accuracy reduction of only 2%. Thus, augmenting the data with synthetic data provides a model agnostic and transparent solution.

### 2. Resources

1. [Presentation slides](https://docs.google.com/presentation/d/1Qc-9QVkUEInTeGwaflN6aTs6hAGx7xB2DUy4CLbheJI/edit#slide=id.p)

### 3. Run on your machine

#### 3.1 Requirements
- Anaconda
- Python 3.7

#### 3.2 Installation steps

1. Clone the github repo to your local machine:
```
git clone https://github.com/krishgn/DataGenDebias
```
2. cd to the repo directory
```
cd DataGenDebias
```
3. Create a virtual environment by:
```
conda env create -f environment.yml
```
4. Activate the virtual environment
```
conda activate tgan_env
```

Now you are ready to run the scripts!

#### 3.3 Running the codes

The scripts can be found in source/ and are as follows:
1. **data_clean.py** : This script cleans the raw data and does some exploratory survey. It creates 3 output csv files.
  i) Clean full dataset
  ii) Clean male only dataset
  iii) Clean female only dataset
2. **data_classify.py** : This script runs the nn classifier on the data and analyses the results. Calculates accuracy, bias metric, etc.
3. **create_TGAN_model.py** : This script takes in the clean data and trains the TGAN on it to create a model
4. **generate_TGAN_data.py** : This scripts takes in the already trained TGAN model and generates the requisite number of synthetic data samples
5. **data_operations.py** : This script contains the different functions used in data_classify

### 4. Data Handling
