'''
This file combines the Label file provided by the dataset and filepaths of the images.

Code to combine labels provided by the dataset with either raw extracted frames or processed images.

A csv with the following columns is formed: 
ClipID,Filepath,Boredom,Engagement,Confusion,Frustration

Filepath will contain raw frames file path or processed image filepath
Author: Team 1
'''

import os
import pandas as pd


raw_labels_file = "Labels/AllLabels.csv"
img_dir = "DataSet"
processed_df_fname = "CSVs/raw_image_labelled.csv"

# This file needs to be re-run after the Processed folder containing the zero-matrix images have been created.
# Uncomment the following 2 lines for combining processed images filepaths
# img_dir = "Processed"
# processed_df_fname = "Processed.csv"

# reading the alllabels.csv from the raw dataset
df = pd.read_csv(raw_labels_file)
# splitting the video names and removing file format
df["ClipID"] = df["ClipID"].apply(lambda id: id.split(".")[0])
df.set_index("ClipID", inplace=True)


mydict = {"ClipID": [], "Filepath": []}
# walking the processed directory to gather all the paths of the processed frames
for path, directories, files in os.walk(img_dir):
    for file in files:
        if file.endswith(".png"):
            # clip id = filename - fileformat - last 4 letters
            # clip id as same as ClipID from raw dataset
            clipId = file.split(".")[0][:-4]
            filepath = os.path.join(path, file)
            mydict["ClipID"].append(clipId)
            mydict["Filepath"].append(filepath)


newdf = pd.DataFrame(mydict, columns=["ClipID", "Filepath"])
newdf.set_index("ClipID", inplace=True)
# joining both datasets in ClipID
newdf = newdf.join(df, on="ClipID", validate="many_to_one")
newdf = newdf.rename(columns=lambda x: x.strip())
newdf.to_csv(processed_df_fname)
print(f"Saved CSV: {processed_df_fname}")
