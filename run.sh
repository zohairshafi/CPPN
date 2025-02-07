#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=cppn
#SBATCH --mem=16G
#SBATCH --partition=netsi_gpu
#SBATCH --nodelist=c4026
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=80:00:00

conda activate mgpu; /home/shafi.z/mambaforge/envs/mgpu/bin/python /home/shafi.z/CPPN/cppn.py

 
