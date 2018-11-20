set -e
rm -rf xgboost
git clone https://github.com/RAMitchell/xgboost.git --recursive
cd xgboost
git checkout multi-gpu-pv
sh ../get_nccl.sh
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DUSE_AVX=ON -DUSE_NCCL=ON -DNCCL_ROOT=$PWD/../nccl
make -j4
cd ..
cd python-package/
python setup.py install --user
cd ../..
