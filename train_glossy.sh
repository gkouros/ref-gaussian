#!/bin/bash
set -e

export TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib64/:$CONDA_PREFIX/bin/:$CONDA_PREFIX/lib/:$CONDA_PREFIX/lib/stub:$LD_LIBRARY_PATH"
export LDFLAGS="-L$CONDA_PREFIX/lib/stubs -L$CONDA_PREFIX/lib64/stubs"

python train.py -s data/glossy_synthetic/angel --eval --white_background   
python train.py -s data/glossy_synthetic/potion --eval  --white_background   
python train.py -s data/glossy_synthetic/horse --eval  --white_background   
python train.py -s data/glossy_synthetic/luyu --eval  --white_background    
python train.py -s data/glossy_synthetic/teapot --eval  --white_background 
python train.py -s data/glossy_synthetic/bell --eval  --white_background   
python train.py -s data/glossy_synthetic/tbell --eval  --white_background  --lambda_normal_smooth 1.0
python train.py -s data/glossy_synthetic/cat --eval  --white_background