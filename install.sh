#!/bin/bash
set -e

conda create -n ref-gaussian python=3.9 -y
conda activate ref-gaussian
conda install -c nvidia/label/cuda-11.7.0 cuda-toolkit cuda-compiler cuda-nvcc cudatoolkit-dev cuda-driver-dev cuda-nvrtc-dev
conda install -c conda-forge gcc_linux-64=11.2.0 gxx_linux-64=11.2.0
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

export TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib64/:$CONDA_PREFIX/bin/:$CONDA_PREFIX/lib/:$CONDA_PREFIX/lib/stub:$LD_LIBRARY_PATH"

pip install submodules/diff-surfel-rasterization_refgaussian
pip install submodules/cubemapencoder
pip install submodules/simple-knn
pip install submodules/raytracing