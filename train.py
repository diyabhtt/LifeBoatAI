import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf
print(f"Found TF-DF {tfdf.__version__}")


train_df = pd.read_csv("/Users/diyabhattarai/Kaggle/input/titanic/train.csv")
train_df = pd.read_csv("/Users/diyabhattarai/Kaggle/input/titanic/test.csv")
print(train_df.head(10))