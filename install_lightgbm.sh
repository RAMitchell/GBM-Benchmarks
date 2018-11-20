#!/bin/sh

set -e

lightgbm_commit=a694712b7fb86cd532eea2c1781b58d4ba58436a

rm -rf LightGBM
git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
git checkout ${lightgbm_commit}
mkdir build ; cd build
cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
make -j$(nproc)
cd ../python-package
python setup.py install --precompile --user
cd ..
