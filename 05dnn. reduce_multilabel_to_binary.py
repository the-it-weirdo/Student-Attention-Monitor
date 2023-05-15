import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

import re


x_regex = r"x: ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$"
y_regex = r"y: ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$"
z_regex = r"z: ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$"


def extract_xyz(inp):
    try:
        x_matches = re.findall(x_regex, inp, re.MULTILINE)
        y_matches = re.findall(y_regex, inp, re.MULTILINE)
        z_matches = re.findall(z_regex, inp, re.MULTILINE)

        data = [(np.double(x[0]), np.double(y[0]), np.double(z[0]))
                for x, y, z in zip(x_matches, y_matches, z_matches)]
        return np.asarray(data)
    except:
        print(inp)
        print(type(inp))
        return np.nan


def reduce_label(inp):
    try:
        if len(inp) == 1 and inp[0] == "Engaged":
            return "Engaged"
        else:
            return "Not Engaged"
    except:
        print(f"Error for label: {inp}")
        return np.nan


paths = glob.glob("Mediapipe_vector/*.csv")

df_list = [pd.read_csv(path) for path in tqdm(paths)]

df = pd.concat(df_list, ignore_index=True)

print(df)
print(df.shape)

df = df[["Mediapipe Output", "label"]]

tqdm.pandas()

df['Mediapipe Output'] = df["Mediapipe Output"].progress_apply(
    lambda x: extract_xyz(x))

df["label"] = df["label"].progress_apply(lambda x: reduce_label(eval(x)))

df.dropna(inplace=True, ignore_index=True)


df.to_csv("Data.csv", index=False)

array = df.to_numpy()

np.save("Data.npy", array)
