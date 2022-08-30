import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm


def normalize(data):
    '''
    This function will normalize the input data to be between 0 and 1

    params:
        data (List) : The list of values you want to normalize

    returns:
        The input data normalized between 0 and 1
    '''
    min_val = min(data)
    if min_val < 0:
        data = [x + abs(min_val) for x in data]
    max_val = max(data)
    return [x/max_val for x in data]


def ohe(df, enc_col):
    '''
    This function will one hot encode the specified column and add it back
    onto the input dataframe

    params:
        df (DataFrame) : The dataframe you wish for the results to be appended to
        enc_col (String) : The column you want to OHE

    returns:
        The OHE columns added onto the input dataframe
    '''

    ohe_df = pd.get_dummies(df[enc_col])
    ohe_df.reset_index(drop=True, inplace=True)
    return pd.concat([df, ohe_df], axis=1)


class CBRecommend():
    def __init__(self, df):
        self.df = df

    def cosine_sim(self, v1, v2):
        '''
        This function will calculate the cosine similarity between two vectors
        '''
        return sum(dot(v1, v2)/(norm(v1)*norm(v2)))

    def recommend(self, book_id, n_rec):
        """
        df (dataframe): The dataframe
        song_id (string): Representing the song name
        n_rec (int): amount of rec user wants
        """

        # calculate similarity of input book_id vector w.r.t all other vectors
        inputVec = self.df.loc[book_id].values
        self.df['sim'] = self.df.apply(
            lambda x: self.cosine_sim(inputVec, x.values), axis=1)

        # returns top n user specified books
        return self.df.nlargest(columns='sim', n=n_rec)


if __name__ == '__main__':
    # constants
    PATH = 'data.csv'

    # import data
    df = pd.read_csv(PATH)

    # normalize the num_pages, ratings, price columns
    df['num_pages_norm'] = normalize(df['num_pages'].values)
    df['book_rating_norm'] = normalize(df['book_rating'].values)
    df['book_price_norm'] = normalize(df['book_price'].values)

    # OHE on publish_year and genre
    df = ohe(df=df, enc_col='publish_year')
    df = ohe(df=df, enc_col='book_genre')
    df = ohe(df=df, enc_col='text_lang')

    # drop redundant columns
    cols = ['publish_year', 'book_genre', 'num_pages',
            'book_rating', 'book_price', 'text_lang']
    df.drop(columns=cols, inplace=True)
    df.set_index('book_id', inplace=True)

    # ran on a sample as an example
    t = df.copy()
    cbr = CBRecommend(df=t)
    print("\nTop 5 Recommendatios")
    print(cbr.recommend(book_id=t.index[0], n_rec=5))
