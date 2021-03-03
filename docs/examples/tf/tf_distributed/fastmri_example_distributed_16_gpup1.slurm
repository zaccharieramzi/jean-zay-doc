#!/bin/bash
#SBATCH --job-name=fastmri_tf_distributed_16_gpup1     # job name
#SBATCH --nodes=4                 # number of nodes
#SBATCH --ntasks-per-node=1         # number of MPI task per node
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=40          # since nodes have 40 cpus
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block  # distribution, might be better to have contiguous blocks
#SBATCH --time=03:00:00             # job length
#SBATCH --output=fastmri_tf_distr_log_16_gpup1_%j.out  # std out
#SBATCH --error=fastmri_tf_distr_log_16_gpup1_%j.out   # std err
#SBATCH --exclusive                 # we reserve the entire node for our job
#SBATCH --qos=qos_gpu-t3         # we are submitting a test job
#SBATCH -C v100-32g


unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

set -x
cd ${SLURM_SUBMIT_DIR}

module purge
module load tensorflow-gpu/py3/2.4.0
export FASTMRI_DATA_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/
export TMP_DIR=$JOBSCRATCH/
export SINGLECOIL_TRAIN_DIR=singlecoil_train/singlecoil_train

srun python ./fastmri_example.py -b 16
