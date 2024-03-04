import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def import_data(path):
    #Converts .csv data to Pandas dataframe
    df = pd.read_csv(path)

    return df

def clean_data(df):
    #Converts all text to lowercase
    df['Text'] = df['Text'].str.lower()
    #Removes excess whitespace
    df['Text'] = ' '.join(df['Text'].str.split())

    #Installs nltk punkt
    nltk.download('punkt')
    #Tokenizes words in data
    df['Text'] = df['Text'].apply(word_tokenize)
        
    #Installs nltk stopwords
    nltk.download('stopwords')
    #Removes stopwords from data
    def remove_stopwords(text):
        result = []
        for word in text:
            if not word in stopwords.words('english'):
                result.append(word)
        
        return result

    df['Text'] = df['Text'].apply(remove_stopwords)

    #Removes punctuation from data
    df['Text'] = RegexpTokenizer(r'\w+').tokenize(' '.join(df['Text']))

    #Installs lemmatization tools
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    #Lemmatizes data
    def lemmatize(text):
        result = []

        for token, pos in pos_tag(text):
            #Identify part of speech
            pos = pos[0].lower()
            #Default to noun if not identified
            if pos not in ['a', 'n', 'v', 'r']:
                pos = 'n'
            #Lemmatize token
            result.append(WordNetLemmatizer().lemmatize(token, pos))
        
        return result
    
    df['Text'] = df['Text'].apply(lemmatize)

    #Removes one-character words
    def remove_short(text):
        result = []

        for word in text:
            if len(word) > 1:
                result.append(word)
        
        return result
    
    df['Text'] = df['Text'].apply(remove_short)

    #Converts from word tokens back to string
    df['Text'] = [' '.join(map(str, token)) for token in df['Text']]

    return df



            




        