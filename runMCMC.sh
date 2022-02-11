#!/bin/bash
#
#SBATCH --job-name=MCMC
#SBATCH --time=360:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=serc
#SBATCH --constraint="CLASS:SH3_CPERF"

module purge
. /oak/stanford/schools/ees/share/serc_env.sh
module load anaconda/3
python3 /scratch/users/taiyi/synthetic-seismograms/inversion.py

