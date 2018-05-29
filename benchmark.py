import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import ml_dataset_loader.datasets as data_loader
import numpy as np

random_seed = 0


class Data:
    def __init__(self, X, y, name, task, train_size=0.6, validation_size=0.2, test_size=0.2):
        assert (train_size + validation_size + test_size) == 1.0
        self.name = name
        self.task = task
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                test_size=test_size,
                                                                                random_state=random_seed)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                              self.y_train,
                                                                              test_size=validation_size / train_size,
                                                                              random_state=random_seed)

        assert (self.X_train.shape[0] + self.X_val.shape[0] + self.X_test.shape[0]) == X.shape[0]

def eval(data, pred):
    if data.task == "Regression":
        return np.sqrt(mean_squared_error(data.y_test, pred))

def train_xgboost(data, params):
    dtrain = xgb.DMatrix(data.X_train, data.y_train)
    dval = xgb.DMatrix(data.X_val, data.y_val)
    dtest = xgb.DMatrix(data.X_test, data.y_test)
    bst = xgb.train(params, dtrain, 10,[(dtrain,"train"),(dval,"val")])

    pred = bst.predict(dtest)
    metric = eval(data,pred)
    print(metric)

num_rows=10

X, y = data_loader.get_synthetic_regression(num_rows=num_rows)
data = Data(X, y, "synthetic", "Regression")
train_xgboost(data, {})
