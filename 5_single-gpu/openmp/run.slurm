#!/bin/bash --login 
#SBATCH --tasks=1 
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --partition=gpuq 
#SBATCH --time=00:05:00 
#SBATCH --export=NONE
module purge
module load cascadelake
module load clang
# Compile
echo "Compiling code"
srun --export=all -u -n 1 make -f makefile.clang clean
srun --export=all -u -n 1 make -f makefile.clang
# Run
echo
echo "Running code"
srun --export=all -u -n 1 ./laplace_mp 4000
