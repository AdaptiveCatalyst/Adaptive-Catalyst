import autograd.numpy as np
import pandas as pd
import scipy.sparse as sparse


def load_data(path='datasets/plays.tsv'):
    raw_data = pd.read_table('datasets/plays.tsv')
    raw_data = raw_data.drop(raw_data.columns[1], axis=1)
    raw_data.columns = ['user', 'artist', 'plays']
    data = raw_data.dropna().sample(frac=1)[:70]

    data['user_id'] = data['user'].astype("category").cat.codes
    data['artist_id'] = data['artist'].astype("category").cat.codes
    data = data.drop(['user', 'artist'], axis=1)

    data = data.loc[data.plays != 0]

    users = list(np.sort(data.user_id.unique()))
    artists = list(np.sort(data.artist_id.unique()))
    plays = list(data.plays)

    rows = data.user_id.astype(int)
    cols = data.artist_id.astype(int)

    return sparse.csr_matrix((plays, (rows, cols)), shape=(len(users), len(artists)))