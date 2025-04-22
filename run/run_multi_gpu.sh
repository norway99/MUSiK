#!/bin/bash

# MUSiK Multi-GPU Script
# This script automates running simulations across multiple GPUs on a single node
# 
# Usage: bash run_multi_gpu.sh <experiment_path> <num_gpus> <workers_per_gpu>
# Example: bash run_multi_gpu.sh my_simulation 4 3

PATH_TO_EXP=$1
NUM_GPUS=$2
WORKERS=$3

if [ -z "$PATH_TO_EXP" ] || [ -z "$NUM_GPUS" ]; then
    echo "Usage: bash run_multi_gpu.sh <experiment_path> <num_gpus> [workers_per_gpu]"
    echo "  experiment_path: Path to the saved experiment directory"
    echo "  num_gpus: Number of GPUs to use"
    echo "  workers_per_gpu: (Optional) Number of CPU workers per GPU (default: 3)"
    exit 1
fi

# Set default workers if not provided
if [ -z "$WORKERS" ]; then
    WORKERS=3
fi

echo "Starting simulation for experiment: $PATH_TO_EXP"
echo "Using $NUM_GPUS GPUs with $WORKERS workers per GPU"

for i in $(seq 0 $((NUM_GPUS-1))); do
    echo "Launching process for GPU $i..."
    if [ $i -eq $((NUM_GPUS-1)) ]; then
        # Run the last process in the foreground to keep the script alive
        python parallel.py -p $PATH_TO_EXP -n $i -s $NUM_GPUS -g 1 -r 1 -w $WORKERS
    else
        # Run other processes in the background
        python parallel.py -p $PATH_TO_EXP -n $i -s $NUM_GPUS -g 1 -r 1 -w $WORKERS &
    fi
done 