import argparse
import time
import ml_dataset_loader.datasets as data_loader
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rows', type=int, default=None,
                        help='Max rows to benchmark for each dataset.')
    parser.add_argument('--num_rounds', type=int, default=500, help='Boosting rounds.')
    args = parser.parse_args()

    X,y = data_loader.get_airline(num_rows = args.rows)

    params = {'objective':'binary:logistic', 'tree_method':'gpu_hist'}
    dtrain = xgb.DMatrix(X, y)

    times = []
    for n_gpus in range(1,9):
        print("Running XGBoost with {} GPUs ...".format(n_gpus))
        params['n_gpus'] = n_gpus
        start = time.time()
        bst = xgb.train(params, dtrain, args.num_rounds)
        times.append( time.time() - start)
        del bst

    print(times)

if __name__ == "__main__":
    main()
