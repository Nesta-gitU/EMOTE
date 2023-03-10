# EMOTE Oversampling Code

The **Embeddings based Synthetic Minority Oversampling TEchnique (EMOTE)** repository contains code for the `oversample()` function, which is used to make a DataFrame balanced by oversampling one of the categories. This function can be used for text classification tasks where the data is imbalanced, and you want to increase the number of samples in the minority class.

The oversampling is performed using bag of words (BOW) representation of the documents. The routine selects words at random and replaces them with words that have embeddings which are close in vector space. This is done for a certain number of randomly selected documents in the original set. The embeddings are trained on the same BOW document set, which ensures that the words exchanged are similar in the context of the specific document set we are dealing with.

The `change_percentage` hyperperameter dictates how much of the original document is changed when creating new documents in the oversampling procedure. 

## Clone the repository

```
git clone https://github.com/Nesta-gitU/EMOTE.git
```

## Requirements

To use this code, you will need to have the following Python packages installed:

- `fasttext`
- `numpy`
- `pandas`
- `nltk`

## Usage

To use the `oversample()` function, you will need to provide it with two input DataFrames: `df` and `y`. `df` should be a DataFrame containing one row with the **preprocessed, tokenized, and stored in lists** text for each document. `y` should be a DataFrame containing one row with the classification for each document. This classification should be binary, with `1` representing the category that you want to oversample, and `0` representing all other categories.

Here's an example of how to use the `oversample()` function:

```python
import pandas as pd
from EMOTE import oversample

# Load the data into DataFrames
df = pd.read_csv("data.csv", header=None)
y = pd.read_csv("labels.csv", header=None)

# Oversample the minority class
oversampled_df, oversampled_y = oversample(df, y, change_percentage=30)

# Save the oversampled data to new files
oversampled_df.to_csv("oversampled_data.csv", header=None, index=None)
oversampled_y.to_csv("oversampled_labels.csv", header=None, index=None)
```

The `oversample()` function returns two DataFrames: `oversampled_df` and `oversampled_y`. `oversampled_df` is the same as the input `df`, but with the oversampled rows added below the original dataframe. `oversampled_y` is also the same as the input `y`, but with the categories of the oversampled docs added below the original categories.

## Acknowledgements
This code is build on the text classification and word embeddings code provided in the [fastText](https://github.com/facebookresearch/fastText) toolkit. We are grateful to the authors of this toolkit at FACEBOOK for making their code available under an open-source license.

## License
This code is released under the MIT License. See LICENSE for more information.
