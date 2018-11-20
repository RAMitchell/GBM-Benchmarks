export CUDA_SHORT=9.0
wget -nc https://developer.download.nvidia.com/compute/redist/nccl/v2.2/nccl_2.2.13-1+cuda${CUDA_SHORT}_x86_64.txz
tar xf "nccl_2.2.13-1+cuda${CUDA_SHORT}_x86_64.txz"
mv "nccl_2.2.13-1+cuda${CUDA_SHORT}_x86_64" nccl
