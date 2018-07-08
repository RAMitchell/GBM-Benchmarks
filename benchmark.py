import argparse
import sys
import time

sys.path.insert(0, 'catboost/catboost/python-package')
import ml_dataset_loader.datasets as data_loader
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

# Global parameters
random_seed = 0
max_depth = 6
learning_rate = 0.1
min_split_loss = 0
min_weight = 1
l1_reg = 0
l2_reg = 1


class Data:
    def __init__(self, X, y, name, task, metric, train_size=0.6, validation_size=0.2,
                 test_size=0.2):
        assert (train_size + validation_size + test_size) == 1.0
        self.name = name
        self.task = task
        self.metric = metric
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                test_size=test_size,
                                                                                random_state=random_seed)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                              self.y_train,
                                                                              test_size=validation_size / train_size,
                                                                              random_state=random_seed)

        assert (self.X_train.shape[0] + self.X_val.shape[0] + self.X_test.shape[0]) == X.shape[0]


def eval(data, pred):
    if data.metric == "RMSE":
        return np.sqrt(mean_squared_error(data.y_test, pred))
    elif data.metric == "Accuracy":
        # Threshold prediction if binary classification
        if data.task == "Classification":
            pred = pred > 0.5
        elif data.task == "Multiclass classification":
            if pred.ndim > 1:
                pred = np.argmax(pred, axis=1)
        return accuracy_score(data.y_test, pred)
    else:
        raise ValueError("Unknown metric: " + data.metric)


def add_data(df, algorithm, data, elapsed, metric):
    time_col = (data.name, 'Time(s)')
    metric_col = (data.name, data.metric)
    try:
        df.insert(len(df.columns), time_col, '-')
        df.insert(len(df.columns), metric_col, '-')
    except:
        pass

    df.at[algorithm, time_col] = elapsed
    df.at[algorithm, metric_col] = metric


def configure_xgboost(data, use_gpu, args):
    params = {'max_depth': max_depth,
              'learning_rate': learning_rate, 'n_gpus': args.n_gpus, 'min_split_loss': min_split_loss,
              'min_child_weight': min_weight, 'alpha': l1_reg, 'lambda': l2_reg, 'debug_verbose':args.debug_verbose}
    if use_gpu:
        params['tree_method'] = 'gpu_hist'
    else:
        params['tree_method'] = 'hist'

    if data.task == "Regression":
        params["objective"] = "reg:linear"
        if use_gpu:
            params["objective"] = "gpu:" + params["objective"]
    elif data.task == "Multiclass classification":
        params["objective"] = "multi:softmax"
        params["num_class"] = np.max(data.y_test) + 1
    elif data.task == "Classification":
        params["objective"] = "binary:logistic"
        if use_gpu:
            params["objective"] = "gpu:" + params["objective"]
    else:
        raise ValueError("Unknown task: " + data.task)

    return params


def configure_lightgbm(data, use_gpu):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'max_depth': max_depth,
        'num_leaves': 2 ** 8,
        'learning_rate': learning_rate, 'min_data_in_leaf': 0,
        'min_sum_hessian_in_leaf': 1, 'lambda_l2': 1, 'min_split_gain': min_split_loss,
        'min_child_weight': min_weight, 'lambda_l1': l1_reg, 'lambda_l2': l2_reg}

    if use_gpu:
        params["device"] = "gpu"

    if data.task == "Regression":
        params["objective"] = "regression"
    elif data.task == "Multiclass classification":
        params["objective"] = "multiclass"
        params["num_class"] = np.max(data.y_test) + 1
    elif data.task == "Classification":
        params["objective"] = "binary"
    else:
        raise ValueError("Unknown task: " + data.task)

    return params


def configure_catboost(data, use_gpu):
    dev_arr = [i for i in range(0, int(args.n_gpus))] if args.n_gpus > 0 else [0]
    params = {'learning_rate': learning_rate, 'depth': max_depth, 'l2_leaf_reg': l2_reg, 'devices' : dev_arr}
    if use_gpu:
        params['task_type'] = 'GPU'
    if data.task == "Multiclass classification":
        params['loss_function'] = 'MultiClass'
        params["classes_count"] = np.max(data.y_test) + 1
        params["eval_metric"] = 'MultiClass'
    return params


def run_xgboost(data, params, args):
    dtrain = xgb.DMatrix(data.X_train, data.y_train)
    dval = xgb.DMatrix(data.X_val, data.y_val)
    dtest = xgb.DMatrix(data.X_test, data.y_test)
    start = time.time()
    bst = xgb.train(params, dtrain, args.num_rounds, [(dtrain, "train"), (dval, "val")])
    elapsed = time.time() - start
    pred = bst.predict(dtest)
    metric = eval(data, pred)
    return elapsed, metric


def train_xgboost(alg, data, df, args):
    if alg not in args.algs:
        return
    use_gpu = True if 'gpu' in alg else False
    params = configure_xgboost(data, use_gpu, args)
    elapsed, metric = run_xgboost(data, params, args)
    add_data(df, alg, data, elapsed, metric)


def run_lightgbm(data, params, args):
    import lightgbm as lgb
    lgb_train = lgb.Dataset(data.X_train, data.y_train)
    lgb_eval = lgb.Dataset(data.X_test, data.y_test, reference=lgb_train)
    start = time.time()
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=args.num_rounds,
                    valid_sets=lgb_eval)
    elapsed = time.time() - start
    pred = gbm.predict(data.X_test)
    metric = eval(data, pred)
    return elapsed, metric


def train_lightgbm(alg, data, df, args):
    if alg not in args.algs:
        return
    use_gpu = True if 'gpu' in alg else False
    params = configure_lightgbm(data, use_gpu)
    elapsed, metric = run_lightgbm(data, params, args)
    add_data(df, alg, data, elapsed, metric)


def run_catboost(data, params, args):
    import catboost as cat
    cat_train = cat.Pool(data.X_train, data.y_train)
    cat_test = cat.Pool(data.X_test, data.y_test)
    cat_val = cat.Pool(data.X_val, data.y_val)

    params['iterations'] = args.num_rounds

    if data.task is "Regression":
        model = cat.CatBoostRegressor(**params)
    else:
        model = cat.CatBoostClassifier(**params)

    start = time.time()
    model.fit(cat_train, use_best_model=False, eval_set=cat_val)
    elapsed = time.time() - start

    if data.task == "Multiclass classification":
        preds = model.predict_proba(cat_test)
    else:
        preds = model.predict(cat_test)

    metric = eval(data, preds)
    return elapsed, metric


def train_catboost(alg, data, df, args):
    if alg not in args.algs:
        return
    use_gpu = True if 'gpu' in alg else False

    # catboost GPU does not work with multiclass
    if data.task == "Multiclass classification" and use_gpu:
        add_data(df, alg, data, 'N/A', 'N/A')
        return

    params = configure_catboost(data, use_gpu)
    elapsed, metric = run_catboost(data, params, args)
    add_data(df, alg, data, elapsed, metric)


class Experiment:
    def __init__(self, data_func, name, task, metric):
        self.data_func = data_func
        self.name = name
        self.task = task
        self.metric = metric

    def run(self, df, args):
        X, y = self.data_func(num_rows=args.rows)
        data = Data(X, y, self.name, self.task, self.metric)
        train_xgboost('xgb-cpu-hist', data, df, args)
        train_xgboost('xgb-gpu-hist', data, df, args)
        train_lightgbm('lightgbm-cpu', data, df, args)
        train_lightgbm('lightgbm-gpu', data, df, args)
        train_catboost('cat-cpu', data, df, args)
        train_catboost('cat-gpu', data, df, args)


experiments = [
    Experiment(data_loader.get_year, "YearPredictionMSD", "Regression", "RMSE"),
    Experiment(data_loader.get_synthetic_regression, "Synthetic", "Regression", "RMSE"),
    Experiment(data_loader.get_higgs, "Higgs", "Classification", "Accuracy"),
    Experiment(data_loader.get_cover_type, "Cover Type", "Multiclass classification", "Accuracy"),
    Experiment(data_loader.get_bosch, "Bosch", "Classification", "Accuracy"),
    Experiment(data_loader.get_airline, "Airline", "Classification", "Accuracy"),
]


def write_results(df, filename, format):
    if format == "latex":
        tmp_df = df.copy()
        tmp_df.columns = pd.MultiIndex.from_tuples(tmp_df.columns)
        with open(filename, "w") as file:
            file.write(tmp_df.to_latex())
    elif format == "csv":
        with open(filename, "w") as file:
            file.write(df.to_csv())
    else:
        raise ValueError("Unknown format: " + format)

    print(format + " results written to: " + filename)


def main():
    all_dataset_names = ''
    for exp in experiments:
        all_dataset_names += exp.name + ','
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rows', type=int, default=None,
                        help='Max rows to benchmark for each dataset.')
    parser.add_argument('--num_rounds', type=int, default=500, help='Boosting rounds.')
    parser.add_argument('--datasets', default=all_dataset_names, help='Datasets to run.')
    parser.add_argument('--debug_verbose', type=int, default=1)
    parser.add_argument('--n_gpus', type=int, default=-1)
    parser.add_argument('--algs', default='xgb-cpu-hist,xgb-gpu-hist,lightgbm-cpu,lightgbm-gpu,'
                                          'cat-cpu,cat-gpu', help='Boosting algorithms to run.')
    args = parser.parse_args()
    df = pd.DataFrame()
    for exp in experiments:
        if exp.name in args.datasets:
            exp.run(df, args)
            # Write partial results at each iteration in case of failure
            print(df.to_string())
            write_results(df, "results.latex", "latex")
            write_results(df, "results.csv", "csv")


if __name__ == "__main__":
    main()
