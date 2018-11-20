#!/bin/sh

set -e

catboost_commit=4b78de61140c301244804f9b74f42774be9af346
lightgbm_commit=f660b5fe4148fc3f5985b10c1fea969b695cb8de

pyver="$(python -V 2>&1 | grep -Eo ' [0-9]\.' | grep -Eo '[0-9]')"

sh install_xgboost.sh

# Catboost
rm -rf catboost
git clone https://github.com/catboost/catboost.git
cd catboost
git checkout ${catboost_commit}
cd catboost/python-package/catboost
../../../ya make -r -DUSE_ARCADIA_PYTHON=no -DOS_SDK=local -DPYTHON_CONFIG=/usr/bin/python${pyver}-config -DCUDA_ROOT=$(dirname $(dirname $(which nvcc)))
cd ../../../..

# LightGBM
rm -rf LightGBM
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
git checkout ${lightgbm_commit}
mkdir build ; cd build
cmake -DUSE_GPU=1 ..
make -j$(nproc)
cd ../python-package
python setup.py install --precompile --user
cd ..
