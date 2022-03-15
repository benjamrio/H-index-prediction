# H-index Prediction Data Challenge submission
The goal of my work is to predict authors h-indexes based on coauthorships links as well as some of their abstract contents. The model uses state of the art text encoding technic as well as a Graph Neural Network.

## Structure of our work
### Scripts
Data extraction and preprocessing has been implemented through main function of 4 scripts : abstracts_to_df.py, abstracts_encoding.py, author_encoding.py, feature_array.py.
The script named main.py calls sequentially these scripts.


### Notebooks
The final model is implemented in a notebook named Node_regression.ipynb. We recommend the use of a virtual environnement to prevent version conflicts for other projects.
Two other notebooks have been left to show some of the research done on the data, that has been directly or undirectly used in the other files.

### Data
Input, intermediary and output data is gathered in this folder. For the scripts to work, the files abstracts.txt, author_papers.txt and coauthorship.edgelist have to be inside this folder. The notebook works with only the features_arry.csv.

### Submissions
Submissions files (csv file with first column author id of the test data and the second column is their predicted h index)

## Requirements
Several well maintained packages have been used :
* gensim
* pytorch
* pytorch.geometric
* pandas
* scikit-network
