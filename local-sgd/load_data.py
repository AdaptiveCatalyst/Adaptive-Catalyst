import numpy as np
import pandas as pd


def one_hot(df, df_cols):
    
    df_1 = adult_data = df.drop(columns = df_cols, axis = 1)
    df_2 = pd.get_dummies(df[df_cols])
    
    return (pd.concat([df_1, df_2], axis=1, join='inner'))


def load_adult(path='datasets/adult.csv', sparsity_treshold=0.2):
    adult = pd.read_csv(path)
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']

    adult.reset_index(inplace = True, drop = True)

    adult = adult[(adult.workclass.values != '?')].copy()
    adult_labels = adult.workclass
    adult.drop(columns=['workclass'], inplace = True)

    adult['marital-status'] = adult['marital-status'].astype('category')
    adult['relationship'] = adult['relationship'].astype('category')
    adult['native-country'] = adult['native-country'].astype('category')
    adult['race'] = adult['race'].astype('category')
    adult['occupation'] = adult['occupation'].astype('category')
    adult['education'] = adult['education'].astype('category')

    gender = {'Male': 1, 'Female': 0}
    adult.gender = [gender[item] for item in adult.gender]

    income = {'<=50K': 0, '>50K': 1}
    adult.income = [income[item] for item in adult.income]

    adult['occupation'] = adult['occupation'].cat.remove_categories('?')
    adult['native-country'] = adult['native-country'].cat.remove_categories('?')

    adult = one_hot(adult, adult.select_dtypes('category').columns)

    adult["age"] = adult["age"] / np.max(adult["age"])
    adult["capital-gain"] = adult["capital-gain"] / np.max(adult["capital-gain"])
    adult["hours-per-week"] = adult["hours-per-week"] / np.max(adult["hours-per-week"])
    adult["educational-num"] = adult["educational-num"] / np.max(adult["educational-num"])
    adult["fnlwgt"] = adult["fnlwgt"] / np.max(adult["fnlwgt"])
    adult["capital-loss"] = adult["capital-loss"] / np.max(adult["capital-loss"])
    
    x = adult.to_numpy()
    y = (adult_labels == "Private").astype(int).to_numpy() * 2 - 1

    m, n = x.shape
    
    sparse_features = [i for i in range(n) if (x[:, i] == 0).mean() >= sparsity_treshold]
    not_sparse_features = [i for i in range(n) if (x[:, i] == 0).mean() < sparsity_treshold]
    
    return m, n, x, y, sparse_features, not_sparse_features