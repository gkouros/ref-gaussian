#!/bin/bash
set -e

export TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib64/:$CONDA_PREFIX/bin/:$CONDA_PREFIX/lib/:$CONDA_PREFIX/lib/stub:$LD_LIBRARY_PATH"
export LDFLAGS="-L$CONDA_PREFIX/lib/stubs -L$CONDA_PREFIX/lib64/stubs"

# eval glossy synthetic data
python eval.py --white_background --save_images --model_path logs/glossy_synthetic/angel
python eval.py --white_background --save_images --model_path logs/glossy_synthetic/bell
python eval.py --white_background --save_images --model_path logs/glossy_synthetic/cat
python eval.py --white_background --save_images --model_path logs/glossy_synthetic/horse
python eval.py --white_background --save_images --model_path logs/glossy_synthetic/luyu
python eval.py --white_background --save_images --model_path logs/glossy_synthetic/potion
python eval.py --white_background --save_images --model_path logs/glossy_synthetic/tbell
python eval.py --white_background --save_images --model_path logs/glossy_synthetic/teapot
# eval ref shiny data
python eval.py --white_background --save_images --model_path logs/ref_shiny/ball
python eval.py --white_background --save_images --model_path logs/ref_shiny/car
python eval.py --white_background --save_images --model_path logs/ref_shiny/coffee
python eval.py --white_background --save_images --model_path logs/ref_shiny/helmet
python eval.py --white_background --save_images --model_path logs/ref_shiny/teapot
python eval.py --white_background --save_images --model_path logs/ref_shiny/toaster
# eval ref real data
python eval.py --white_background --save_images --model_path logs/ref_real/gardenspheres
python eval.py --white_background --save_images --model_path logs/ref_real/sedan
python eval.py --white_background --save_images --model_path logs/ref_real/toycar
# eval nerf synthetic data
python eval.py --white_background --save_images --model_path logs/nerf_synthetic/chair
python eval.py --white_background --save_images --model_path logs/nerf_synthetic/drums
python eval.py --white_background --save_images --model_path logs/nerf_synthetic/ficus
python eval.py --white_background --save_images --model_path logs/nerf_synthetic/hotdog
python eval.py --white_background --save_images --model_path logs/nerf_synthetic/lego
python eval.py --white_background --save_images --model_path logs/nerf_synthetic/materials
python eval.py --white_background --save_images --model_path logs/nerf_synthetic/mic
python eval.py --white_background --save_images --model_path logs/nerf_synthetic/ship