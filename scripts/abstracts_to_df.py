import pandas as pd
import json
import os
import re
from string import punctuation
import ast

data_path = os.path.join(os.getcwd(), "data")
print(data_path)


def preprocess_word(text):
    """Clean a word : remove punctuation, lowercase, remove extra blank characters

    Arguments:
        text {string} -- text to be processed, often a word

    Returns:
        text {string} -- cleaned input
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(punctuation)}]", "", text)
    text = " ".join(text.split())
    return text


def main():
    """
    From the raw file abstracts.txt, create a csv file named abstracts_cleaned.csv
    with columns abstract_id, length and wordlist, easier to read for later use.
    Run in approx. 6 minutes.
    """
    abstracts = pd.read_csv(
        os.path.join(data_path, "abstracts.txt"), delimiter="\n", header=None
    )

    abstracts[["abstract_id", "abstract"]] = abstracts[0].str.split(
        "----", 1, expand=True
    )
    abstracts.drop(0, axis=1, inplace=True)

    abstracts["length"] = abstracts["abstract"].apply(
        lambda x: json.loads(x)["IndexLength"]
    )

    abstracts["InvertedIndex"] = abstracts["abstract"].apply(
        lambda x: json.loads(x)["InvertedIndex"]
    )
    abstracts.drop(columns="abstract", inplace=True)

    abstracts["word_list"] = abstracts["InvertedIndex"].apply(
        lambda x: list(
            set(preprocess_word(word) for word in ast.literal_eval(str(x)).keys())
        )
    )
    abstracts.drop(columns="InvertedIndex", inplace=True)

    print(abstracts.head())
    abstracts.to_csv(os.path.join(data_path, "abstracts_cleaned.csv"))


if __name__ == "__main__":
    main()
