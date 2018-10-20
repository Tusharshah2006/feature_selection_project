# %load q05_forward_selected/build.py
# Default imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()


# Your solution code here
def forward_selected(df, LinReg):
    features = df.drop('SalePrice', axis=1)
    target = df['SalePrice']
    feature_list = list(features.columns)
    best_features = []
    best_scores = []
    
    while len(feature_list) > 0:
        scores_with_features = []
        
        for feature in feature_list:
            best_features.append(feature)

            LinReg.fit(features[best_features], target)
            rsquare = LinReg.score(features[best_features], target)          
            scores_with_features.append((rsquare, feature))
            
            best_features.remove(feature)
            
        scores_with_features.sort()
        best_score, best_candidate = scores_with_features.pop()
        
        feature_list.remove(best_candidate)
        
        best_features.append(best_candidate)
        best_scores.append(best_score)

    return best_features, best_scores 
    
forward_selected(data, model)


