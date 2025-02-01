import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf
print(f"Found TF-DF {tfdf.__version__}")


train_df = pd.read_csv("C:\Users\sharm\Documents\LifeBoat-AI\train.csv")
test_df = pd.read_csv("C:\Users\sharm\Documents\LifeBoat-AI\test.csv")
print(train_df.head(10))