#!/bin/bash

# Recommended to call this script from the wrapper function run.sh

# Navigate to the run directory
cd /my/path/to/MUSiK/run

# Load and activate musik conda environment (this should be prepared in advance)
source /my/path/to/anaconda/etc/profile.d/conda.sh
conda activate musik

# Load CUDA and other modules (if needed - will depend on your compute cluster)
module load cuda/11.2
module load glibc/2.35
module load zlib/1.2.11
module load hdf5/1.10.9

echo "SLURM_ARRAY_TASK_ID: " ${SLURM_ARRAY_TASK_ID}

# Run the parallel script
python parallel.py -p $1 -n ${SLURM_ARRAY_TASK_ID} -s $2 -g $3 -r $4 -w $5