#!/bin/sh

set -e

xgboost_commit=286dccb8e85b053cebc829e06f018dce184306d9
catboost_commit=27294dbe0b72a33f3655b2da88c771616781c062
lightgbm_commit=f660b5fe4148fc3f5985b10c1fea969b695cb8de

# XGBoost
rm -rf xgboost
git clone https://github.com/dmlc/xgboost.git --recursive
cd xgboost
git checkout ${xgboost_commit}
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DUSE_NCCL=ON -DUSE_AVX=ON
make -j4
cd ..
cd python-package/
python setup.py install
cd ../..

# Catboost
rm -rf catboost
git clone https://github.com/catboost/catboost.git
cd catboost
git checkout ${catboost_commit}
cd catboost/python-package/catboost
../../../ya make -r -DUSE_ARCADIA_PYTHON=no -DOS_SDK=local -DPYTHON_CONFIG=/usr/bin/python3-config
cd ../../../..

# LightGBM
rm -rf LightGBM
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
git checkout ${lightgbm_commit}
mkdir build ; cd build
cmake -DUSE_GPU=1 ..
make -j$(nproc)
cd ../python-package
python setup.py install --precompile
cd ..
