from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
import lightgbm as lgb

from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# For Model training
class LGBMTrain(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, df):
        # Splitting of the dataset to train-test
        X = df.drop('binaryRiskStatus', axis=1)
        y = df['binaryRiskStatus']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
        
        params = { 
            'objective': 'binary', 
            'boosting_type': 'gbdt',
            'eval_metric': 'auc',
            'num_leaves': 31, 
            'learning_rate': 0.05,
        } 
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
        cm_display.plot()
        plt.show()
        
        return model