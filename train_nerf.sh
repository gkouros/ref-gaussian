#!/bin/bash
set -e

export TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib64/:$CONDA_PREFIX/bin/:$CONDA_PREFIX/lib/:$CONDA_PREFIX/lib/stub:$LD_LIBRARY_PATH"
export LDFLAGS="-L$CONDA_PREFIX/lib/stubs -L$CONDA_PREFIX/lib64/stubs"

python train.py -s data/nerf_synthetic/chair --eval --white_background
python train.py -s data/nerf_synthetic/drums --eval --white_background
python train.py -s data/nerf_synthetic/ficus --eval --white_background
python train.py -s data/nerf_synthetic/hotdog --eval --white_background
python train.py -s data/nerf_synthetic/lego --eval --white_background
python train.py -s data/nerf_synthetic/materials --eval --white_background
python train.py -s data/nerf_synthetic/mic --eval --white_background
python train.py -s data/nerf_synthetic/ship --eval --white_background

python eval.py --white_background --save_images --model_path data/nerf_synthetic/chair
python eval.py --white_background --save_images --model_path data/nerf_synthetic/drums
python eval.py --white_background --save_images --model_path data/nerf_synthetic/ficus
python eval.py --white_background --save_images --model_path data/nerf_synthetic/hotdog
python eval.py --white_background --save_images --model_path data/nerf_synthetic/lego
python eval.py --white_background --save_images --model_path data/nerf_synthetic/materials
python eval.py --white_background --save_images --model_path data/nerf_synthetic/mic
python eval.py --white_background --save_images --model_path data/nerf_synthetic/ship