######## imports #########
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
##########################

'''
deliver a dataframe where the column is called allText.
'''
def preprocess(df_all):
    stopwords1 = stopwords.words('english')
    # the words we write here should be based on anlysis from bag of words I think
    new_stopwords = ["reuters", "said", 'reuter']
    stopwords1.extend(new_stopwords)

    stopwords2 = set(stopwords1)

    for i in range(0, len(df_all)):
        doc = df_all.loc[i, 'allText']
        doc_tokens = word_tokenize(doc)
        doc_words = [word.lower() for word in doc_tokens if word.isalpha()]
        doc_nostop = [w for w in doc_words if not w in stopwords2]

        for index, j in enumerate(doc_nostop):
            doc_nostop[index] = WordNetLemmatizer().lemmatize(j)

        df_all.loc[i, 'allText'] = doc_nostop

    return df_all


def dummy_fun(doc):
    return doc


def get_tf_idf_matrix(series_train, series_test, series_val):
    list_train = series_train.tolist()
    list_test = series_test.tolist()
    list_val = series_val.tolist() 

    # preprocess data
    df1_all = preprocess(list_train)
    df2_all = preprocess(list_test)
    df3_all = preprocess(list_val)

    # create a bow matrix
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None)

    series1 = df1_all['text'].T
    list1 = series1.tolist()

    series2 = df2_all['text'].T
    list2 = series2.tolist()

    series3 = df3_all['text'].T
    list3 = series3.tolist()

    # use only transform on test dataset to use same vocab as training set
    X1_all = vectorizer.fit_transform(list1)
    X2_all = vectorizer.transform(list2)
    X3_all = vectorizer.transform(list3)

    df1_final = pd.DataFrame(
        X1_all.toarray(), columns=vectorizer.get_feature_names_out())
    df2_final = pd.DataFrame(
        X2_all.toarray(), columns=vectorizer.get_feature_names_out())
    df3_final = pd.DataFrame(
        X3_all.toarray(), columns=vectorizer.get_feature_names_out())

    return (df1_final, df2_final, df3_final)
