######## Imports ########
import pandas as pd
from sklearn.model_selection import train_test_split
from ReadFile import all_data_to_dataframe
from TextProcessing import get_tf_idf_matrix
#########################
def main():
    print('running')
    path = r"C:/Nesta/School/jaar 2/dsp/reuters-dataset/sgm-files" 

    df = all_data_to_dataframe(path)

    ###############split data into training, validation and testing subsets #############
    df_train, df_test, df_val, y_train, y_test, y_val = train_test_val_split(df, 'trade')

    df_train_all_text = pd.DataFrame(df_train['allText'], columns= ['allText'])
    print(df_train_all_text)

    ################################ oversample here ####################################

    #####################################################################################

    tfidf_train, tfidf_val, tfidf_test = get_tf_idf_matrix(df_train['allText'], df_val['allText'], df_test['allText'])




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