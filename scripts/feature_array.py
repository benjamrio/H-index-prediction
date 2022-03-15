import networkx as nx
import os
import pandas as pd
import csv

data_path = os.path.join(os.getcwd(), "data")
print(data_path)


def main():
    authors_encoding = pd.read_csv(os.path.join(data_path), "authors_encoding.csv")
    g = nx.read_edgelist("coauthorship.edgelist", create_using=nx.Graph(), nodetype=int)
    mean_vector = authors_encoding[["feature_" + str(i) for i in range(50)]].mean(
        axis=0
    )
    mean_vector
    features_cols = ["feature_" + str(i) for i in range(50)]
    authors_list = authors_encoding["author"].unique()
    features_array = [
        authors_encoding[authors_encoding["author"] == author][
            features_cols
        ].values.tolist()
        if int(author) in authors_list
        else mean_vector
        for author in g
    ]
    with open("/content/drive/MyDrive/Hindex/features_array.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=",")
        csvWriter.writerows(features_array)

