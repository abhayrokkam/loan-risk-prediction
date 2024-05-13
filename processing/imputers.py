from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

mf_imputer = Pipeline(
    steps=[(
        'imputer', SimpleImputer(strategy='most_frequent')
        )])

avg_imputer = Pipeline(
    steps=[(
        'imputer', SimpleImputer(strategy='mean')
        )])