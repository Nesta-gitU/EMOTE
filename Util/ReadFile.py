from bs4 import BeautifulSoup
import pandas as pd
import os


def data_to_list(final_list, f):
    data_raw = f.read()

    soup = BeautifulSoup(data_raw, 'html.parser')
    articles = soup.findAll('reuters')
    n_articles = len(articles)

    for article in articles:
        if article.title == None or article.body == None or len(article.topics) == 0:
            continue
        ## you can access the tags with article.title, article.body etc.
        inner_list = []
        inner_list.append(article.date.string)
        inner_list.append(article.topics.find_all(string = True))
        inner_list.append(article.title.string.lower())
        inner_list.append(article.body.string.lower())
        
        
        final_list.append(inner_list)

def all_data_to_dataframe(path):
    # Change the directory
    os.chdir(path)

    final_list = []

    # Iterate over all the files in the directory
    for file in os.listdir():
        if file.endswith('.sgm'):
            # Create the filepath of particular file
            file_path =f"{path}/{file}"
        
            f = open(file_path, mode= 'r' , encoding = 'utf−8' , errors = 'ignore')
            data_to_list(final_list, f)
        
    
    df = pd.DataFrame(final_list)
    df.columns = ['date', 'topics', 'title', 'body']

    # add column with combined title and body text
    df['allText'] = df['title'] + ' ' + df['body']
    
    return df.reset_index(drop=True)

    

def test():
    vijf = 5
    zes = 6

    vijf = zes
    vijf = 4

    

    print(zes)

test()