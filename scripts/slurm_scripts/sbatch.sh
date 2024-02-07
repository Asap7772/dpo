#!/bin/bash
#SBATCH --job-name="sft-hh"
#SBATCH --account=bcfp-delta-gpu
#SBATCH --mail-user=ftajwar@cs.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpuA40x4
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 11:00:00
#SBATCH --output=/scratch/bcfp/asingh15/dpo/slurm_logs/ppo_exps_%j.out # Standard output and error log

bash /scratch/bcfp/asingh15/dpo/scripts/run_sft.sh $1