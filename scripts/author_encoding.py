import pandas as pd
import numpy as np
import os

data_path = os.path.join(os.getcwd(), "data")


def main():
    """Give each author the mean of the vectors of its abstracts, 
    in order to obtain an aggregated feature vector for each author.
    """

    authors = pd.read_csv(
        os.path.join(data_path, "authors_papers"), delimiter="\n", header=None
    )
    authors[["author", "texts"]] = authors[0].str.split(":", 1, expand=True)
    authors.drop(0, axis=1, inplace=True)
    authors["texts"] = (
        authors["texts"].str.split("-", 5).apply(lambda x: sorted(list(set(x))))
    )
    authors["nb_articles"] = authors["texts"].apply(lambda x: x.count("-") + 1)
    for i in range(5):
        authors["text_" + str(i)] = authors["texts"].apply(
            lambda x: x[i] if i < len(x) else None
        )
    authors.drop("texts", axis=1, inplace=True)
    authors["author"] = authors["author"].astype("int64")

    abstract_authors = pd.DataFrame(columns={"abstract", "author"})
    for i in range(5):
        authorships = (
            authors[[f"text_{i}", "author"]]
            .rename(columns={"text_" + str(i): "abstract"})
            .dropna(subset=["abstract"], inplace=False)
        )
        abstract_authors = pd.concat([abstract_authors, authorships])
    assert len(abstract_authors) == np.sum(
        [authors["text_" + str(i)].count() for i in range(5)]
    )
    abstract_authors = abstract_authors.astype({"abstract": "int64"})

    abstracts_encoding = pd.read_csv(
        os.path.join(data_path, "abstracts_encoding.csv"), index_col=0
    )
    abstract_authors = abstract_authors.merge(abstracts_encoding, on="abstract")
    abstract_authors[["feature_" + str(i) for i in range(50)]] = abstract_authors[
        ["feature_" + str(i) for i in range(50)]
    ].astype("float32")

    author_encoding = abstract_authors.groupby("author").mean()[
        ["feature_" + str(i) for i in range(50)]
    ]
    author_encoding.to_csv(os.path.join(data_path, "author_encoding.csv"))


if __name__ == "__main__":
    main()
