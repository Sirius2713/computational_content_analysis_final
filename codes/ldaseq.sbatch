#!/bin/sh

#SBATCH --job-name=dynamic-topic-modeling
#SBATCH --account=soci40133
#SBATCH --partition=caslake
#SBATCH --ntasks-per-node=1  # number of tasks
#SBATCH --cpus-per-task=1    # number of threads per task
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=16GB
#SBATCH --out=dtm.out

# LOAD MODULES
module load python

# DO COMPUTE WORK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u dtm.py > dtm.out
