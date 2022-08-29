import fasttext
import csv
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def oversample(df, y):
    """calculates the sum of x and y.     
    Parameters:     
        df (DataFrame): a (number of docs, 1) DataFrame containing one row with the **preprocessed, tokenized and stored in lists** text for each document. 
        y (Dataframe): a (number of docs, 1) DataFrame containing one row with the classification for each of the docs. Should be binary. 
                       1 for the category that we are oversampling, 0 for all other categories. 
    Returns:     
       DataFrame (DataFrame): A dataframe with the oversampled rows added below the original dataframe.
       y (DataFrame): a (number of docs + number of new oversampled docs, 1) DataFrame that adds the categories of the oversampled docs below the original categories. 
    """   

    oversampled_list = []

    for row in df:
        
        oversampled_list.append(row)

    oversampled_df = pd.DataFrame(oversampled_list)
    


    hallo = 5



def process_fasttext(df, y):
    """Take the dataframes with input and convert them into the text file format which fastText accepts.     
    Parameters:     
        df (DataFrame): a (number of docs, 1) DataFrame containing one row with the **preprocessed and tokenized** text for each document. 
        y (Dataframe): a (number of docs, 1) DataFrame containing one row with the classification for each of the docs. Should be binary. 
                       1 for the category that we are oversampling, 0 for all other categories. 
    Returns:     
        **this function does not return anything**

    Action:
        After executing this function there will be a text file stored in fasttext_text.txt
    """   
 
    result = pd.concat([y, df], axis=1, join='inner') 
    
    result.iloc[:, 0] = result.iloc[:, 0].apply(lambda x: '__label__' + str(x))

    result.iloc[:, 1] = result.iloc[:, 1].apply(lambda x: ' '.join(x))

    result.set_axis(['category', 'questions'], axis=1, inplace=True)

    print(result)

    result[['category', 'questions']].to_csv('E-MOTE/fasttext_text.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")


