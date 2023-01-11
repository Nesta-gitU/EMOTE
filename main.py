######## Imports ########
import pandas as pd
from sklearn.model_selection import train_test_split
from Util.ReadFile import all_data_to_dataframe
from Util.TextProcessing import get_tf_idf_matrix
from Util.TextProcessing import preprocess
from EMOTE_directory.src.E_MOTE.E_MOTE import oversample
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import LinearSVC
from imblearn import under_sampling, over_sampling


#########################
def main():
    
    path = r"C:/Nesta/School/jaar 2/dsp/reuters-dataset/sgm-files" 

    df, topic_list = all_data_to_dataframe(path)

    print(topic_list)
    print(len(topic_list))

    rest_of_topics = ['dkr', 'jobs', 'acq', 'cpu', 'rape-oil', 'corn', 'zinc', 'ship', 'rice', 'rapeseed', 'soybean', 'strategic-metal', 'f-cattle', 'livestock', 'nzdlr', 'hk', 'fishmeal', 'naphtha', 'cocoa', 'austdlr', 'groundnut', 'retail', 'cornglutenfeed', 'citruspulp', 'sorghum', 'platinum', 'stg', 'meal-feed', 'instal-debt', 'oilseed', 'palm-oil', 'gas', 'sunseed', 'grain', 'saudriyal', 'hog', 'dmk', 'cottonseed', 'plywood', 'nickel', 'iron-steel', 'lead', 'dfl', 'wpi', 'coconut-oil', 'jet', 'lei', 'sun-oil', 'gnp', 'tin', 'peseta', 'skr', 'red-bean', 'palmkernel', 'castorseed', 'soy-oil', 'money-fx', 'yen', 'copra-cake', 'lin-oil', 'potato', 'rand', 'castor-oil', 'palladium', 'copper', 'earn', 'rubber', 'coffee', 'inventories', 'ipi', 'orange', 'lit', 'dlr', 'veg-oil', 'sun-meal', 'tea']

    for topic in rest_of_topics:
        get_output_3methods(topic, df)
    

def get_output_3methods(topic, df):
    print('#########################output for ', topic,'############################')

    df = df.copy()

    ###############split data into training, validation and testing subsets #############
    return_tuple = train_test_val_split(df, topic)

    if return_tuple == False:
        print('execution stopped because minority class has less than 3 docs')
        return  
    
    df_train, df_test, df_val, y_train, y_test, y_val = return_tuple

    df_train_all_text = pd.DataFrame(df_train.copy()['allText'], columns= ['allText'])

    ################################ preprocess below ####################################

    prep_df_train_all_text = preprocess(df_train_all_text)

    ################################ undersample below ####################################
    rus = under_sampling.RandomUnderSampler(sampling_strategy = {0 : 5000}, random_state=0)
    df_train_under, y_train_under_array = rus.fit_resample(prep_df_train_all_text, y_train['topics'].tolist())

    y_train_under = pd.DataFrame(y_train_under_array, columns = ['topics'])

    ################################ oversample below ####################################

    #RANDOM
    rus_over = over_sampling.RandomOverSampler(sampling_strategy = 1.0, random_state=0)
    df_train_random, y_train_random_array = rus_over.fit_resample(df_train_under, y_train_under['topics'].tolist())

    y_train_random = pd.DataFrame(y_train_random_array, columns = ['topics'])

    #E-MOTE
    print('E-mote start')
    oversampled_df_train_all_text, oversampled_y = oversample(df_train_under, y_train_under) 

    ################################ do tfidf svm below ###################################
    # preprocess validation and test data. And preprocess non oversampled train data
    df_train = preprocess(df_train)
    df_test = preprocess(df_test)
    df_val = preprocess(df_val)

    # get tfidf matrices
    tfidf_train_over, tfidf_val_over, tfidf_test_over = get_tf_idf_matrix(oversampled_df_train_all_text, df_val, df_test)
    tfidf_train_random, tfidf_val_random, tfidf_test_random = get_tf_idf_matrix(df_train_random, df_val, df_test)
    tfidf_train_under, tfidf_val_under, tfidf_test_under = get_tf_idf_matrix(df_train_under, df_val, df_test)


    tfidf_train, tfidf_val, tfidf_test = get_tf_idf_matrix(df_train, df_val, df_test)

    # apply smote to the tfidf matrices 
    rus_smote = over_sampling.SMOTE(sampling_strategy = 1.0, random_state=0)
    try:
        tfidf_train_smote, y_train_array_smote = rus_smote.fit_resample(tfidf_train_under, y_train_under['topics'].tolist())
    except:
        print('execution stopped because of error with smote')
        return

    y_train_smote = pd.DataFrame(y_train_array_smote, columns = ['topics'])

    # perform svm 
    #print('#################### not oversampled ################')
    #svm_predict(tfidf_train, tfidf_test, tfidf_val, y_train, y_test, y_val)

    print('#################### oversampled E-MOTE ####################')
    svm_predict(tfidf_train_over, tfidf_test_over, tfidf_val_over, oversampled_y, y_test, y_val)

    print('#################### oversampled smote ####################')
    svm_predict(tfidf_train_smote, tfidf_test_under, tfidf_val_under, y_train_smote, y_test, y_val)

    print('#################### oversampled random ####################')
    svm_predict(tfidf_train_random, tfidf_test_random, tfidf_val_random, y_train_random, y_test, y_val)  


def svm_predict(tfidf_train, tfidf_test, tfidf_val, y_train, y_test, y_val):
    clf = LinearSVC()
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


#get approximately 8/1/1 split with train/test/val
def train_test_val_split(df, selected_topic):

    number_y_1_docs = 0
    
    for i in range(len(df['topics'])):
        if selected_topic in df.loc[i, 'topics']:
            df.at[i, 'topics'] = 1
            number_y_1_docs = number_y_1_docs + 1

        else: df.at[i, 'topics'] = 0

    ######## check if we have at least 3 y=1 classes
    print('number of minority class docs: ', number_y_1_docs)

    if(number_y_1_docs < 3):
        return False
    #####################################################
    
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