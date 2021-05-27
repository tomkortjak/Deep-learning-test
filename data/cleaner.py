from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import sklearn
import re
import pandas as pd


class CleaningMethods:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def prep_reviews(self, df):
        """
        This method cleans the review dataframe to make it ready for sentiment analysis.
        :return: Cleaned dataframe
        """
        # Rename columns
        df.columns = ['Hotel_Name', 'Positive_Review', 'Negative_Review', 'Reviewer_Score', 'Amount_reviews',
                      'Average_Score', 'lat', 'lng']
        # Replace empty columns with np.nan and drop them
        df['Negative_Review'].replace('', np.nan, inplace=True)
        df['Positive_Review'].replace('', np.nan, inplace=True)
        df.dropna(inplace=True)

        # Make all review texts lowercase
        df['Negative_Review'] = df['Negative_Review'].str.lower()
        df['Negative_Review'] = df['Negative_Review'].str.strip()
        df['Positive_Review'] = df['Positive_Review'].str.lower()
        df['Positive_Review'] = df['Positive_Review'].str.strip()

        # Cast the score types to float
        df['Reviewer_Score'] = df['Reviewer_Score'].astype(float)
        df['Average_Score'] = df['Average_Score'].astype(float)

        # Remove punctuation
        df["Positive_Review"] = df['Positive_Review'].str.replace('[^\w\s]', '')
        df["Negative_Review"] = df['Negative_Review'].str.replace('[^\w\s]', '')

    def split_reviews(self, df):
        # Initiate dataframes
        df_positive = pd.DataFrame()
        df_negative = pd.DataFrame()

        # Split dataframes based on positive and negative columns
        df_positive['review'] = df['Positive_Review']
        df_positive['label'] = 1
        df_negative['review'] = df['Negative_Review']
        df_negative['label'] = 0

        # Combine Dataframes back into one and shuffle it for better analysis results
        df_combined = pd.concat([df_positive, df_negative])
        df_shuffled = sklearn.utils.shuffle(df_combined, random_state=20)
        return df_shuffled

    def get_stemmed_text(self, df):
        # Take stem of all words in reviews
        df['review'] = df['review'].apply(
            lambda x: " ".join([self.stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split()]).lower())
