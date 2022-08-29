import fasttext
import csv
from numpy import number
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from random import sample

def oversample(df, y, percentage_docs_to_add):
    """calculates the sum of x and y.     
    Parameters:     
        df (DataFrame): a (number of docs, 1) DataFrame containing one row with the **preprocessed, tokenized and stored in lists** text for each document. 
        y (Dataframe): a (number of docs, 1) DataFrame containing one row with the classification for each of the docs. Should be binary. 
                       1 for the category that we are oversampling, 0 for all other categories.            
        percentage_docs_to_add: In a percentage of the current number of documents in our category. How many extra documents of our category should be added by oversampling.
    Returns:     
       DataFrame (DataFrame): A dataframe with the oversampled rows added below the original dataframe.
       y (DataFrame): a (number of docs + number of new oversampled docs, 1) DataFrame that adds the categories of the oversampled docs below the original categories. 
    """
    # check if input is valid 
    if percentage_docs_to_add <= 0: raise Exception("The pecentage of extra documents to add should be higher than 0")

    # initialize fastText model and data
    process_fasttext(df, y)
    model = fasttext.train_supervised('fasttext_text.txt')
    
    # create a seperate df with only category 1 to select from when oversampling
    df_1 = df[y.iloc[:, 0] == 1]
    df_1.reset_index(drop=True, inplace=True)
    print(df_1)

    # create a list with the random indexes of the documents to be oversampled
    cat1_n = len(df_1)
    wanted_cat1_n = int(cat1_n *(1 + (percentage_docs_to_add/100)))

    number_of_extra_docs_needed = wanted_cat1_n - cat1_n
    index_list = []

    while number_of_extra_docs_needed > 0:
        current_number_of_docs_to_add = number_of_extra_docs_needed

        if(number_of_extra_docs_needed > cat1_n):
            current_number_of_docs_to_add = cat1_n
        
        
        all_index_list = list(range(0, cat1_n))
        print(all_index_list)
        print(current_number_of_docs_to_add)
        print(sample(all_index_list, current_number_of_docs_to_add))

        index_list.append(sample(all_index_list, current_number_of_docs_to_add))

        number_of_extra_docs_needed = number_of_extra_docs_needed - current_number_of_docs_to_add


    # create the oversampled documents
    oversampled_list = []
    for index in index_list:
        current_doc = df_1.iloc[index]
        print(index)

        oversampled_list.append(current_doc)

    oversampled_df = pd.DataFrame(oversampled_list)
    




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

    result[['category', 'questions']].to_csv('fasttext_text.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")


