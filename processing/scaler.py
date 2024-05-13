from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

scaler = Pipeline(
    steps=[(
        'scaler', MinMaxScaler()
        )])