#!/bin/bash
#PBS -P cp23
#PBS -q gpuvolta
#PBS -l ncpus=96
#PBS -l ngpus=8
#PBS -l mem=200GB
#PBS -l walltime=48:00:00
#PBS -l wd
#PBS -l storage=gdata/cp23+scratch/cp23
NNODES=$((PBS_NCPUS / PBS_NCI_NCPUS_PER_NODE)) #96/48
node_gpu=$((PBS_NGPUS / NNODES)) #8/2
NGPUS=${PBS_NGPUS}
cur_host=$(cat $PBS_NODEFILE | head -n 1)
MASTER_NAME="${cur_host/\.gadi\.nci\.org\.au/}"
PORT=29501
dist_url="tcp://$(python3 -c 'import socket; print(socket.gethostbyname(socket.gethostname()))'):$PORT"
echo $dist_url
echo $MASTER_NAME
EXP_DIR="./output/idow_convnext_large_12ep_lvis_iouobj+kl+alleny+detach-cluster"
for inode in $(seq 1 $PBS_NCI_NCPUS_PER_NODE $PBS_NCPUS); do
  if test $inode -eq 1 
  then
    JOB_RANK=0
  else
    JOB_RANK=1
  fi
  pbsdsh -n $inode -v ../tools/run_8gpus.sh $NGPUS $dist_url $PORT $JOB_RANK $node_gpu $EXP_DIR &
done