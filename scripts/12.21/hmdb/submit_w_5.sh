
#!/bin/bash
#PBS -P cp23
#PBS -l ngpus=4
#PBS -l ncpus=48
#PBS -l mem=380GB
#PBS -q gpuvolta
#PBS -l jobfs=32GB
#PBS -l walltime=24:00:00
#PBS -l wd
#PBS -l storage=scratch/dg97+scratch/kf09+gdata/kf09

cd /home/135/qc2666/dg/VideoMAEv2

module load cuda/12.2.2

# Activate Conda
export CONDA_ENV='/scratch/kf09/qc2666/miniconda3/bin/activate'
source $CONDA_ENV videomae

bash scripts/12.21/hmdb/vit_b_w_5_param.sh
