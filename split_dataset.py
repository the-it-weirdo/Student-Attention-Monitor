'''
Code to split the dataset.
We used this to reduce training complexity and tackle the time constraint.
Author: Team 1
'''

from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("Raw_Multilabel.csv")

train, test = train_test_split(df, test_size=0.2)

train.to_csv("Raw_Multilabel80.csv", index=False)
test.to_csv("Raw_Multilabel20.csv", index=False)
