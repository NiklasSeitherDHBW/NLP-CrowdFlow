"""
Configuration file for the project
"""

RANDOM_STATE = 42  # static random state for reproducibility

SENTIMENTS = None  # will be updated via the Utils.load_data() function
FEATURES = ["text",]  # features to be used as predictors for the model
TARGET = "label"  # target variable to predict

TRAIN_SET = "cryptonews_manual.csv"  # training dataset
CV_SET = "ieee_labeled.csv"  # cross-validation dataset

MODEL_DIR = "../res/models/"  # specify the directory to save/load the models to/from