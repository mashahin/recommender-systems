import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def normalize(pred_ratings):
    '''
    This function will normalize the input pred_ratings

    params:
        pred_ratings (List -> List) : The prediction ratings 
    '''
    return (pred_ratings - pred_ratings.min()) / (pred_ratings.max() - pred_ratings.min())


def generate_prediction_df(mat, pt_df, n_factors):
    '''
    This function will calculate the single value decomposition of the input matrix
    given n_factors. It will then generate and normalize the user rating predictions.

    params:
        mat (CSR Matrix) : scipy csr matrix corresponding to the pivot table (pt_df)
        pt_df (DataFrame) : pandas dataframe which is a pivot table
        n_factors (Integer) : Number of singular values and vectors to compute. 
                              Must be 1 <= n_factors < min(mat.shape). 
    '''

    if not 1 <= n_factors < min(mat.shape):
        raise ValueError("Must be 1 <= n_factors < min(mat.shape)")

    # matrix factorization
    u, s, v = svds(mat, k=n_factors)
    s = np.diag(s)

    # calculate pred ratings
    pred_ratings = np.dot(np.dot(u, s), v)
    pred_ratings = normalize(pred_ratings)

    # convert to df
    pred_df = pd.DataFrame(
        pred_ratings,
        columns=pt_df.columns,
        index=list(pt_df.index)
    ).transpose()
    return pred_df


def recommend_items(pred_df, usr_id, n_recs):
    '''
    Given a usr_id and pred_df this function will recommend
    items to the user.

    params:
        pred_df (DataFrame) : generated from `generate_prediction_df` function
        usr_id (Integer) : The user you wish to get item recommendations for
        n_recs (Integer) : The number of recommendations you want for this user
    '''

    usr_pred = pred_df[usr_id].sort_values(
        ascending=False).reset_index().rename(columns={usr_id: 'sim'})
    rec_df = usr_pred.sort_values(by='sim', ascending=False).head(n_recs)
    return rec_df


if __name__ == '__main__':
    # constants
    PATH = 'data.csv'

    # import data
    df = pd.read_csv(PATH)
    print(df.shape)

    # generate a pivot table with readers on the index and books on the column and values being the ratings
    pt_df = df.pivot_table(
        columns='book_id',
        index='reader_id',
        values='book_rating'
    ).fillna(0)

    # convert to a csr matrix
    mat = pt_df.values
    mat = csr_matrix(mat)

    pred_df = generate_prediction_df(mat, pt_df, 10)

    # generate recommendations
    print("\nTop 5 Recommendations")
    print(recommend_items(pred_df, 5, 5))
