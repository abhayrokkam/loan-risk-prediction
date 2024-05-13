import numpy as np 
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

# For Debugging
class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        print(X)
        return X

    def fit(self, X):
        return self

# Transform target variable (Before train-test-split)
class TargetTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):        
        # Dropping null values of target column
        X.dropna(subset=['loanStatus'], inplace=True)       
        
        # Target into Binary
        high_risk = ['Rejected' , 'External Collection' , 'Internal Collection' , 'Charged Off' , 'Settled Bankruptcy']
        
        binaryRiskStatus = ['High' if val in high_risk else 'Low' for val in X['loanStatus']]
        X['binaryRiskStatus'] = binaryRiskStatus

        X = X.drop(columns = ['loanStatus'])
        
        return X

# Outlier removal (before train-test split)
class OutlierRemoval(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for col in ['loanAmount', 'originallyScheduledPaymentAmount', 'apr']:
            Q1 = X[col].quantile(0.2)
            Q3 = X[col].quantile(0.8)   # 20-80 limit
            IQR = Q3 - Q1

            low_bound = Q1 - 1.5 * IQR
            up_bound = Q3 + 1.5 * IQR
            
            X = X[(X[col] > low_bound) & (X[col] < up_bound)]
        
        return X

# Unnecessary data removal (for all columns)
class UnnecessaryDelete(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Dropping unnecessary columns
        X = X.drop(columns = ['loanId', 'anon_ssn', 'originatedDate', 'fpStatus', 'clarityFraudId'])    
        
        return X

# Feature Engineering
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Part 1 - Date
        X['applicationYear'] = pd.DatetimeIndex(X['applicationDate']).year
        X['applicationMonth'] = pd.DatetimeIndex(X['applicationDate']).month
        X = X.drop(columns = ['applicationDate'])
        
        # Part 2 - LC-LA-Ratio
        X['lc_la_ratio'] = X['leadCost'] * X['loanAmount']
        
        # Part 3 - Season
        month_to_season = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        }
        
        X['season'] = X['applicationMonth'].map(month_to_season)
        
        return X
