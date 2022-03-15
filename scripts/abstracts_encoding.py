import gensim
from collections import namedtuple
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import ast
import os

data_path = os.path.join(os.getcwd(), "data")
print(data_path)
abstracts = pd.read_csv(os.path.join(data_path, "abstracts_cleaned.csv"), index_col=0)


def build_encoding():
    # Defining training data
    ROW_LIM = 1000
    df_train = abstracts[["abstract_id", "word_list"]].sample(n=ROW_LIM)
    print(f"Shape of the train dataframe : {df_train.shape}")

    # creating the corpus of documents
    analyzedDocument = namedtuple("AnalyzedDocument", "words tags")
    train_corpus = []
    df_train.apply(
        lambda x: train_corpus.append(
            analyzedDocument(x["word_list"], [x["abstract_id"]])
        ),
        axis=1,
    )

    # implementing the model using doc2vec
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    abstracts["encoding"] = abstracts.apply(
        lambda x: model.infer_vector(ast.literal_eval(x["word_list"])), axis=1
    )
    abstracts["encoding"] = abstracts["encoding"].apply(normalize)

    # apply the model to encode the abstracts
    abstracts_encoding = abstracts[["abstract_id", "length", "encoding"]]
    abstracts_encoding[["feature_" + str(i) for i in range(50)]] = abstracts_encoding[
        "encoding"
    ].str.split(", ", 50, expand=True)
    abstracts_encoding["feature_" + str(0)] = abstracts_encoding[
        "feature_" + str(0)
    ].str[1:]
    abstracts_encoding["feature_" + str(49)] = abstracts_encoding[
        "feature_" + str(49)
    ].str[:-1]
    return abstracts[["abstract_id", "length", "encoding"]]


def normalize(float_list):
    """function to row normalize encoding, inspired by source code of TORCH_GEOMETRIC.TRANSFORMS.NORMALIZE_FEATURES
    Args:
        float_list -- list of floating numbers
    Returns the values divided by the sum of the list (sum of list is now 1)
    """
    s = sum(float_list)
    return [x / s for x in float_list]


def main():
    abstracts_encoding = build_encoding()
    abstracts_encoding.to_csv(os.path.join(data_path, "abstracts_encoding"))


if __name__ == "__main__":
    main()

