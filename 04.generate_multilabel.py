'''
Code to process the labels.
It converts the very low, low, high and very high labels in the raw dataset to multi-class labels.
Author: Team 1
'''
import pandas as pd
import os


processed_dataset = "Processed.csv"
multilabel_dataset = "Multilabel.csv"

# For getting labels with raw images
# processed_dataset = "raw_image_labelled.csv"
# multilabel_dataset = "Raw_Multilabel.csv"


def generate_multilabel_list(bored, engaged, confused, frustrated):
    '''
    Function to create multi-class labels 
    '''
    label = []
    if bored:
        label.append("Bored")
    if engaged:
        label.append("Engaged")
    if confused:
        label.append("Confused")
    if frustrated:
        label.append("Frustrated")
    if len(label) == 0:
        label.append("Unknown")

    return label


df = pd.read_csv(processed_dataset)

# 2 -> high. So if more or equal to 2, image belongs to that class.
df['label'] = df.apply(lambda x: generate_multilabel_list(x["Boredom"] >= 2,
                                                          x["Engagement"] >= 2,
                                                          x["Confusion"] >= 2,
                                                          x["Frustration"] >= 2), axis=1)


print(
    f"Formed the following multi-classes:\n {df['label'].apply(tuple).value_counts()}")
df.to_csv(multilabel_dataset, index=False)
print(f"Saved Multilabel: {multilabel_dataset}")


df["Type"] = df["Filepath"].apply(lambda x: x.split(os.path.sep)[1])

type_df = df.set_index("Type")

print("Train")
print(type_df.loc["Train"]["label"].apply(tuple).value_counts())
print("Test")
print(type_df.loc["Test"]["label"].apply(tuple).value_counts())
print("Validation")
print(type_df.loc["Validation"]["label"].apply(tuple).value_counts())
