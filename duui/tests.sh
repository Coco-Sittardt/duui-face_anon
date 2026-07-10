#!/bin/bash

#SBATCH --partition=gpu_l40s  # Select the partition
#SBATCH --job-name=tests         # Job name
#SBATCH --output=output_%j.log     # Standard output
#SBATCH --error=error_%j.log       # Standard error
#SBATCH --time=01:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --mem=4G                   # Memory requirement
#SBATCH --cpus-per-task=2          # CPU cores per task
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:1                  # Request one GPU

EXPORT HF_TOKEN=""
cd test
mvn test
