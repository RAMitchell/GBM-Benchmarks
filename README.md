# GBM-Benchmarks
GBM benchmark suite for the purpose of evaluating the speed of XGBoost GPU on multi-GPU systems with large datasets.

This benchmark is designed to be run on an [AWS p3.16xlarge](https://aws.amazon.com/ec2/instance-types/p3/) instance with 8 V100 GPUs. It is recommended to use the [Deep Learning Base AMI (Ubuntu)](https://aws.amazon.com/marketplace/pp/B077GCZ4GR) and storage of at least 150GB for this task.

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
Run on 7 June 2018

|              | "('YearPredictionMSD' 'Time(s)')" | "('YearPredictionMSD' 'RMSE')" | "('Synthetic' 'Time(s)')" | "('Synthetic' 'RMSE')" | "('Higgs' 'Time(s)')" | "('Higgs' 'Accuracy')" | "('Cover Type' 'Time(s)')" | "('Cover Type' 'Accuracy')" | "('Bosch' 'Time(s)')" | "('Bosch' 'Accuracy')" | "('Airline' 'Time(s)')" | "('Airline' 'Accuracy')" |
|--------------|-----------------------------------|--------------------------------|---------------------------|------------------------|-----------------------|------------------------|----------------------------|-----------------------------|-----------------------|------------------------|-------------------------|--------------------------|
| xgb-cpu-hist | 397.27372694015503                | 8.879391001888838              | 565.2947809696198         | 13.610471042735508     | 470.09188079833984    | 0.7474345454545455     | 464.05221605300903         | 0.891982134712529           | 752.5890619754791     | 0.994454065469905      | 1948.264995098114       | 0.7494303418939346       |
| xgb-gpu-hist | 34.25581908226013                 | 8.879935744972384              | 38.48715591430664         | 13.460576927868603     | 34.07960486412048     | 0.747475               | 103.3895480632782          | 0.8928685145822397          | 32.12634301185608     | 0.9944244984160507     | 144.8635070323944       | 0.749484266051801        |
| lightgbm-cpu | 38.12508988380432                 | 8.877691075962955              | 421.0538258552551         | 13.585034611136265     | 306.9785330295563     | 0.7473804545454545     | 83.76876091957092          | 0.8928340920630277          | 250.0972819328308     | 0.9943907074973601     | 916.0412080287933       | 0.7504912703697312       |
| lightgbm-gpu | 80.04824590682983                 | 8.88175154521266               | 609.4814240932465         | 13.585007307447382     | 529.5377051830292     | 0.7469995454545455     | 126.52870297431946         | 0.8930578384379061          | 487.14922618865967    | 0.9944076029567054     | 614.7447829246521       | 0.749949160947056        |
| cat-cpu      | 38.49950695037842                 | 8.994799241732066              | 436.58789801597595        | 9.389984249250787      | 397.02287697792053    | 0.7406940909090909     | 288.1107921600342          | 0.8518626885708631          | 242.90423798561096    | 0.9944160506863781     | 2949.0425968170166      | 0.7265709745333714       |
| cat-gpu      | 9.802947044372559                 | 9.036473602545339              | 35.474628925323486        | 9.399963630634538      | 30.145710945129395    | 0.7406177272727272     | N/A                        | N/A                         | N/A                   | N/A                    | 303.35544514656067      | 0.7277047723183877       |
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


