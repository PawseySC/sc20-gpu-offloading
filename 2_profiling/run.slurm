#!/bin/bash --login 
#SBATCH --tasks=1 
#SBATCH --partition=gpuq 
#SBATCH --time=00:05:00 
#SBATCH --export=NONE
module purge
module load cascadelake
module load gcc/9.2.0
# Compile
echo "Compiling code"
srun --export=all -u -n 1 make clean
srun --export=all -u -n 1 make 
# Run
echo
echo "Running code"
srun --export=all -u -n 1 ./laplace 4000
# Generate profiling report
echo
echo "Generating profiling report"
srun --export=all -u -n 1 gprof -lbp laplace gmon.out
