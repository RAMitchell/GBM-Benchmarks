# GBM-Benchmarks
GBM benchmark suite for the purpose of evaluating the speed of XGBoost GPU on multi-GPU systems with large datasets.

This benchmark is designed to be run on an [AWS p3.16xlarge](https://aws.amazon.com/ec2/instance-types/p3/) instance with 8 V100 GPUs. It is recommended to use the [Deep Learning Base AMI (Ubuntu)](https://aws.amazon.com/marketplace/pp/B077GCZ4GR) for this task.

## Requirements
- Cuda 9.0
- Python 3
- Sklearn, Pandas, numpy
- [Kaggle CLI](https://github.com/Kaggle/kaggle-api) with a valid API token
## Usage
Install XGBoost, LightGBM and Catboost:
```sh
sh install_gbm.sh
```

Run the benchmarks
```sh
python3 benchmarks.py

```
It can be useful to run the benchmarks with a small number of rows/rounds to quickly check everything is working:
```sh
python3 benchmarks.py --rows 100 --num_rounds 10
```


Benchmark parameters:
```sh
usage: benchmark.py [-h] [--rows ROWS] [--num_rounds NUM_ROUNDS]
                    [--datasets DATASETS] [--algs ALGS]

optional arguments:
  -h, --help            show this help message and exit
  --rows ROWS           Max rows to benchmark for each dataset. (default:
                        None)
  --num_rounds NUM_ROUNDS
                        Boosting rounds. (default: 500)
  --datasets DATASETS   Datasets to run. (default:
                        YearPredictionMSD,Synthetic,Higgs,Cover
                        Type,Bosch,Airline,)
  --algs ALGS           Boosting algorithms to run. (default: xgb-cpu-
                        hist,xgb-gpu-hist,lightgbm-cpu,lightgbm-gpu,cat-
                        cpu,cat-gpu)

```

## Datasets
Datasets are loaded using [ml_dataset_loader](https://github.com/RAMitchell/ml_dataset_loader/tree/ac520d8c34d1d3bd68819e49dffd97f4a3f671c6). Datasets are automatically downloaded and cached over subsequent runs. Allow time for these downloads on the first run.

## Example results

## Scalability test
We test the scalability of multi-GPU XGBoost by running with between 1-8 GPUs on the airline dataset and timing the results.
```sh
python3 scalability.py -h
usage: scalability.py [-h] [--rows ROWS] [--num_rounds NUM_ROUNDS]

optional arguments:
  -h, --help            show this help message and exit
  --rows ROWS           Max rows to benchmark for each dataset. (default:
                        None)
  --num_rounds NUM_ROUNDS
                        Boosting rounds. (default: 500)
```


