#!/bin/bash

#SBATCH --partition=gpu_h200   # Select the partition
#SBATCH --job-name=face_anon         # Job name
#SBATCH --output=output_%j.log     # Standard output
#SBATCH --error=error_%j.log       # Standard error
#SBATCH --time=03:00:00            # Maximum runtime (HH:MM:SS)
#SBATCH --mem=8G                   # Memory requirement
#SBATCH --cpus-per-task=2          # CPU cores per task
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:1

source /home/staff_homes/sittardt/miniconda3/etc/profile.d/conda.sh

conda activate face-anon-simple


python -m uvicorn main:app --port 8001
