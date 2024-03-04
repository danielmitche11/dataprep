import pandas as pd
import nltk

def import_data(path):
        #Converts .csv data to Pandas dataframe
        df = pd.read_csv(path)

        return df

def clean_data(df):
        #Converts all text to lowercase
        df['Text'] = df['Text'].str.lower()
        #Removes excess whitespace
        df['Text'] = ' '.join(df['Text'].str.split())

        return df




        