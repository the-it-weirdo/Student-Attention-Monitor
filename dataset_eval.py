import pandas as pd
import matplotlib.pyplot as plt

dataset = "Multilabel.csv"


df = pd.read_csv(dataset)


class_counts = df["label"].value_counts()


# for simple bar chart of all classes and their counts.
plt.figure(figsize=(15, 10))
plt.bar_label(plt.bar(class_counts.index, class_counts.values))
# after plotting the data, format the labels
current_values = plt.gca().get_yticks()
# using format string '{:.0f}' here but you can choose others
plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
plt.xticks(rotation=30, ha="right")
plt.subplots_adjust(bottom=0.2)
plt.grid()
plt.title("Instance count for each group of classes")
plt.xlabel("Group of classes")
plt.ylabel("Count of instances")
plt.savefig("multilabel_counts.png")


#####
