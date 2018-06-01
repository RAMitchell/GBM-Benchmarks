#!/bin/sh

set -e

xgboost_commit=b7cbec4d4bf3e102164f083638d41de0aafa2999
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

# Temporarily disabled
: <<'END'
# Catboost
rm -rf catboost
git clone https://github.com/catboost/catboost.git
cd catboost
git checkout ${catboost_commit}
cd catboost/python-package/catboost
../../../ya make -r -DUSE_ARCADIA_PYTHON=no -DOS_SDK=local -DPYTHON_CONFIG=/usr/bin/python3-config
cd ../../../..
END

# LightGBM
rm -rf LightGBM
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
git checkout ${lightgbm_commit}
cd ../python-package
python setup.py install --gpu
