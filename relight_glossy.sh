#!/bin/bash
set -e

# RESCALE_FLAG="--rescale_relight"
RESCALE_FLAG=""

# for scene in "bell" ; do
#     for envmap in "neon" ; do
# for scene in "angel" ; do
#     for envmap in "golf" ; do
for scene in "angel" "bell" "cat" "horse" "luyu" "potion" "tbell" "teapot" ; do
    for envmap in "corridor" "golf" "neon" ; do
        echo "Relighting $scene with $envmap.exr"
        python -u eval.py -m logs/glossy_synthetic/${scene}  --save_images $RESCALE_FLAG --relight_gt_path="data/glossy_synthetic/relight_gt/${scene}_${envmap}" --relight_envmap_path="data/glossy_synthetic/relight_gt/${envmap}.exr"
    done
done
