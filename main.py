######## Imports ########
import pandas as pd
from sklearn.model_selection import train_test_split
from Util.ReadFile import all_data_to_dataframe
from Util.TextProcessing import get_tf_idf_matrix
from Util.TextProcessing import preprocess
from EMOTE.E_MOTE import oversample
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from imblearn import under_sampling, over_sampling


#########################
def main():
    print('running')
    path = r"C:/Nesta/School/jaar 2/dsp/reuters-dataset/sgm-files" 

    df = all_data_to_dataframe(path)

    ###############split data into training, validation and testing subsets #############
    df_train, df_test, df_val, y_train, y_test, y_val = train_test_val_split(df, 'corn')

    df_train_all_text = pd.DataFrame(df_train.copy()['allText'], columns= ['allText'])
    print(df_train_all_text)
    print(y_train)

    ################################ preprocess below ####################################

    prep_df_train_all_text = preprocess(df_train_all_text)
    print(prep_df_train_all_text)

    ################################ undersample below ####################################
    rus = under_sampling.RandomUnderSampler(sampling_strategy = {0 : 5000}, random_state=0)
    df_train_under, y_train_under_array = rus.fit_resample(prep_df_train_all_text, y_train['topics'].tolist())

    y_train_under = pd.DataFrame(y_train_under_array, columns = ['topics'])

    ################################ oversample below ####################################

    #E-MOTE
    oversampled_df_train_all_text, oversampled_y = oversample(df_train_under, y_train_under)
    print(oversampled_df_train_all_text)
    print(oversampled_y) 

    #SMOTE

    #RANDOM

    ################################ do tfidf svm below ###################################
    # preprocess validation and test data. And preprocess non oversampled train data
    df_train = preprocess(df_train)
    df_test = preprocess(df_test)
    df_val = preprocess(df_val)

    # get tfidf matrices
    tfidf_train_over, tfidf_val_over, tfidf_test_over = get_tf_idf_matrix(oversampled_df_train_all_text, df_val, df_test)
    #tfidf_train, tfidf_val, tfidf_test = get_tf_idf_matrix(df_train, df_val, df_test)

    print('oversampled')
    print(tfidf_train_over)
    print(tfidf_test_over)
    print(tfidf_val_over)
    
    print('not oversampled')
    #print(tfidf_train)
    #print(tfidf_test)
    #print(tfidf_val)

    # perform svm 
    #print('#################### not oversampled ################')
    #svm_predict(tfidf_train, tfidf_test, tfidf_val, y_train, y_test, y_val)

    print('#################### oversampled E-MOTE ####################')
    svm_predict(tfidf_train_over, tfidf_test_over, tfidf_val_over, oversampled_y, y_test, y_val)
 


def svm_predict(tfidf_train, tfidf_test, tfidf_val, y_train, y_test, y_val):
    clf = SVC()
    clf.fit(tfidf_train ,y_train['topics'].tolist())
    y_pred = clf.predict(tfidf_val)
    y_pred2 = clf.predict(tfidf_test)

    print('val')
    print('accuracy: ', metrics.accuracy_score(y_val['topics'].tolist(), y_pred))
    print('f1: ', metrics.f1_score(y_val['topics'].tolist(), y_pred))
    print('precision: ', metrics.precision_score(y_val['topics'].tolist(), y_pred))
    print('recall: ', metrics.recall_score(y_val['topics'].tolist(), y_pred))
    
    print('test')
    print('accuracy: ', metrics.accuracy_score(y_test['topics'].tolist(), y_pred2))
    print('f1: ', metrics.f1_score(y_test['topics'].tolist(), y_pred2))
    print('precision: ', metrics.precision_score(y_test['topics'].tolist(), y_pred2))
    print('recall: ', metrics.recall_score(y_test['topics'].tolist(), y_pred2))

def lasso_predict(tfidf_train, tfidf_test, tfidf_val, y_train, y_test, y_val):
    object = LogisticRegression(penalty='l1', solver='liblinear', class_weight="balanced")
    clf = object.fit(tfidf_train, y_train) 

    y_pred = clf.predict(tfidf_val)
    y_pred2 = clf.predict(tfidf_test)

    print('val')
    print('accuracy: ', metrics.accuracy_score(y_val['topics'].tolist(), y_pred))
    print('f1: ', metrics.f1_score(y_val['topics'].tolist(), y_pred))
    print('precision: ', metrics.precision_score(y_val['topics'].tolist(), y_pred))
    print('recall: ', metrics.recall_score(y_val['topics'].tolist(), y_pred))
    
    print('test')
    print('accuracy: ', metrics.accuracy_score(y_test['topics'].tolist(), y_pred2))
    print('f1: ', metrics.f1_score(y_test['topics'].tolist(), y_pred2))
    print('precision: ', metrics.precision_score(y_test['topics'].tolist(), y_pred2))
    print('recall: ', metrics.recall_score(y_test['topics'].tolist(), y_pred2))





#get approximately 8/1/1 split with train/test/val
def train_test_val_split(df, selected_topic):
       
    print(df.tail(5))
    
    for i in range(len(df['topics'])):
        if selected_topic in df.loc[i, 'topics']:
            df.at[i, 'topics'] = 1

        else: df.at[i, 'topics'] = 0

    print(df.tail(5))
    
    df_train, df_test = train_test_split(df, test_size=0.1, random_state= 9, stratify = df['topics'].tolist()) ## random state is 9 keep it like this 
    df_train, df_val = train_test_split(df_train, test_size=(1/9), random_state = 32, stratify = df_train['topics'].tolist())

    df_train.reset_index(drop=True, inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    #### get the dependent variables
    y_train = pd.DataFrame(df_train['topics'], columns = ['topics'])
    y_val = pd.DataFrame(df_val['topics'], columns = ['topics'])
    y_test  = pd.DataFrame(df_test['topics'], columns = ['topics'])

    return df_train, df_test, df_val, y_train, y_test, y_val

if __name__ == "__main__":
    main()