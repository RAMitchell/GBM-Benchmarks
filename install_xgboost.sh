set -e
xgboost_commit=dc2bfbfde144ada58c6992f0315e27064a5f8ea6
rm -rf xgboost
git clone https://github.com/dmlc/xgboost.git --recursive
cd xgboost
git checkout ${xgboost_commit}
sh ../get_nccl.sh
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DUSE_AVX=ON -DUSE_NCCL=ON -DNCCL_ROOT=$PWD/../nccl
make -j4
cd ..
cd python-package/
python setup.py install --user
cd ../..
