#!/bin/bash --login 
#SBATCH --tasks=1 
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --partition=gpuq 
#SBATCH --time=00:05:00 
#SBATCH --export=NONE
module purge
module load cascadelake
module load gcc/9.2.0
# Compile
echo "Compiling code"
srun --export=all -u -n 1 make -f makefile.gcc clean
srun --export=all -u -n 1 make -f makefile.gcc
# Run C code
echo
echo "Running C code"
srun --export=all -u -n 1 ./laplace 4000
# Run Fortran code
echo
echo "Running Fortran code"
srun --export=all -u -n 1 ./laplacef 4000
