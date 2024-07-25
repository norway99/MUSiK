#!/bin/bash

# Recommended to call this script from the wrapper function run.sh

cd /cbica/projects/superres/trevor/repos/musik_dev_private/run

source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
conda activate kwu
module load cuda/11.2
module load glibc/2.35
module load zlib/1.2.11
module load hdf5/1.10.9

echo "SLURM_ARRAY_TASK_ID: " ${SLURM_ARRAY_TASK_ID}

python parallel.py -p $1 -n ${SLURM_ARRAY_TASK_ID} -s $2 -g $3 -r $4 -w $5