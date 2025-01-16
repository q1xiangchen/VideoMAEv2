
#!/bin/bash
#PBS -P kf09
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=30GB
#PBS -q gpuvolta
#PBS -l jobfs=32GB
#PBS -l walltime=03:00:00
#PBS -l wd
#PBS -l storage=scratch/dg97+scratch/kf09+gdata/kf09

cd /home/135/qc2666/dg/VideoMAEv2

module load cuda/12.2.2

# Activate Conda
export CONDA_ENV='/scratch/kf09/qc2666/miniconda3/bin/activate'
source $CONDA_ENV videomae
