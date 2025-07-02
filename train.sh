#!/bin/bash
set -e

export TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib64/:$CONDA_PREFIX/bin/:$CONDA_PREFIX/lib/:$CONDA_PREFIX/lib/stub:$LD_LIBRARY_PATH"
export LDFLAGS="-L$CONDA_PREFIX/lib/stubs -L$CONDA_PREFIX/lib64/stubs"

python train.py -s data/ref_shiny/coffee --eval --white_background   
python train.py -s data/ref_shiny/helmet --eval  --white_background  --lambda_normal_smooth 1.0
python train.py -s data/ref_shiny/ball --eval  --white_background --lambda_normal_smooth 1.0 
python train.py -s data/ref_shiny/teapot --eval  --white_background 
python train.py -s data/ref_shiny/toaster --eval  --white_background   
python train.py -s data/ref_shiny/car --eval  --white_background 

python train.py -s data/glossy_synthetic/angel_blender --eval --white_background   
python train.py -s data/glossy_synthetic/potion_blender --eval  --white_background   
python train.py -s data/glossy_synthetic/horse_blender --eval  --white_background   
python train.py -s data/glossy_synthetic/luyu_blender --eval  --white_background    
python train.py -s data/glossy_synthetic/teapot_blender --eval  --white_background 
python train.py -s data/glossy_synthetic/bell_blender --eval  --white_background   
python train.py -s data/glossy_synthetic/tbell_blender --eval  --white_background  --lambda_normal_smooth 1.0
python train.py -s data/glossy_synthetic/cat_blender --eval  --white_background 


python train.py -s data/ref_real/gardenspheres --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 
python train.py -s data/ref_real/toycar --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4
python train.py -s data/ref_real/sedan --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 8 