#!/bin/bash

# cd to the dir of the script so that you can execute it from anywhere
DIR=$( realpath -e -- $( dirname -- ${BASH_SOURCE[0]}))
cd $DIR
echo $DIR

NUM_JOBS=1
# ARGS=""# condor_send --jobname "rflgs" --queue "$NUM_JOBS" --conda-env 'ref-gaussian' --gpumem 16 --mem 16 --gpus 1 --cpus 4 --timeout 2.99 --nice 1 -c "bash train.sh"
# condor_send --jobname "rflgs" --queue "$NUM_JOBS" --conda-env 'ref-gaussian' --gpumem 16 --mem 16 --gpus 1 --cpus 4 --timeout 2.99 --nice 1 -c "bash train.sh && bash eval.sh"
condor_send --jobname "rflgs" --queue "$NUM_JOBS" --conda-env 'ref-gaussian' --gpumem 16 --mem 16 --gpus 1 --cpus 4 --timeout 2.99 --nice 1 -c "bash eval.sh"

# condor_send --jobname "rflgs" --queue "$NUM_JOBS" --conda-env 'ref-gaussian' --gpumem 16 --mem 32 --gpus 1 --cpus 4 --timeout 2.99 --nice 1 -c "bash train_shiny.sh"
# condor_send --jobname "rflgs" --queue "$NUM_JOBS" --conda-env 'ref-gaussian' --gpumem 16 --mem 32 --gpus 1 --cpus 4 --timeout 2.99 --nice 1 -c "bash train_glossy.sh"
# condor_send --jobname "rflgs" --queue "$NUM_JOBS" --conda-env 'ref-gaussian' --gpumem 24 --mem 32 --gpus 1 --cpus 4 --timeout 2.99 --nice 1 -c "bash train_real.sh"
# condor_send --jobname "rflgs" --queue "$NUM_JOBS" --conda-env 'ref-gaussian' --gpumem 16 --mem 32 --gpus 1 --cpus 4 --timeout 2.99 --nice 1 -c "bash train_nerf.sh"
