import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import ctypes


lib1 = ctypes.cdll.LoadLibrary(
    '/home/andry/.conda/envs/py365/lib/python3.6/site-packages/xgboost/lib/libxgboost.so')

def evalerror(preds, dtrain):
    from sklearn.metrics import mean_squared_error
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    return 'rmse', mean_squared_error(labels, preds)


digits = load_digits(n_class=2)
X = digits['data']
y = digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf1 = xgb.XGBClassifier(learning_rate=0.1)
clf1.fit(X_train, y_train, early_stopping_rounds=5, eval_metric="auc",
         eval_set=[(X_test, y_test)])


cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
            feval=evalerror, maximize=True,
            early_stopping_rounds=1)