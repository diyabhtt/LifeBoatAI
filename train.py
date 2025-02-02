import numpy as np
import pandas as pd
import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf
print(f"Found TF-DF {tfdf.__version__}")

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

train_df = pd.read_csv("/Users/diyabhattarai/LifeBoatAI/train.csv")
test_df = pd.read_csv("/Users/diyabhattarai/LifeBoatAI/test.csv")
#print(train_df.head(5))

#function to clean up the data
def preprocess(df):
    df = df.copy() #this creates copy of the data so that we dont modify the original one
    def normalize_name(x): #helper function inside preprocess (in python u can create helper methods inside other methods)
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])  #splits it up into string array and gets rid of all the unwanted characters and then joins them again
    def ticket_number(x):
        return x.split(" ")[-1] # splits and then  takes last item from list 
    def ticket_item(x):
        items = x.split(" ") #splits
        if len(items) == 1: #if it only has one number
            return "NONE"
        return "_".join(items[0:-1]) #joins everything except last part
    df["Name"] = df["Name"].apply(normalize_name) #accesses name column of df, applies the function to every item in name column 
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)
    return df
preprocessed_train_df = preprocess(train_df)
preprocessed_test_df = preprocess(test_df)
#print(preprocessed_train_df.head(5))

#remove unnecessary columns
input_features = list(preprocessed_train_df.columns) #gets column names, converts into list
input_features.remove("Ticket")
input_features.remove("PassengerId")
input_features.remove("Survived")
#these are not useful for model training so we remove it
print(f"Input features: {input_features}") #f string allows putting variables inside 
          