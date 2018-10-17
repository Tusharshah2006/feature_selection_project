# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df, k=20):
    predictors = df.drop('SalePrice', axis=1)
    target = df['SalePrice']
    
    fs = SelectPercentile(f_regression, k)
    features = fs.fit_transform(predictors, target)
    features_by_score = [predictors.columns[i] for i in np.argsort(fs.scores_)[::-1]]

    return features_by_score[:7]

percentile_k_features(data, 20)

