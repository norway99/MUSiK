#!/bin/bash

# Example usage:
# "bash run.sh --job-name custom_musik_simulation --output outputs/custom_musik_simulation_output_%A_%a.out --dir path/to/experiment/directory --nodes 16 --workers 3 --cuda 1 --time "2-00" --mem "64G" --repeat 0"

# Default values for command-line arguments
Job_name="musik_simulation"
Directory="./"
Nodes=1
Workers=4
Cuda=1
Time="4-00"
Mem="64G"
Repeat=1
Output="outputs/musik_simulation_%A_%a.out"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --job-name) Job_name="$2"; shift ;;
        --dir) Directory="$2"; shift ;;
        --nodes) Nodes="$2"; shift ;;
        --workers) Workers="$2"; shift ;;
        --cuda) Cuda="$2"; shift ;;
        --time) Time="$2"; shift ;;
        --mem) Mem="$2"; shift ;;
        --repeat) Repeat="$2"; shift ;;
        --output) Output="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

sbatch --job-name=$Job_name --output=$Output --array="0-$((Nodes-1))" --ntasks=1 --cpus-per-task=$Workers --gpus=1 --mem-per-gpu=$Mem --time=$Time musik_submit.sh $Directory $Nodes $Cuda $Repeat $Workers