#!/bin/bash
#SBATCH -J G21.5-0.9
#SBATCH -t 96:00:00
#SBATCH --ntasks 101
#SBATCH -p parallel
#SBATCH -o job.%J.out

#modules
module purge
module load anaconda
source activate astropython
#module load openmpi
#sh clear.bat
mpirun python mcmc_bleeding_edge.py

