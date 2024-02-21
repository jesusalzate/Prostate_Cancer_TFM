#!/bin/bash
#SBATCH --job-name=nnUNet_Prostate_4

#SBATCH --partition=gpuceib

#SBATCH --cpus-per-task 7

#SBATCH --mem 50G

#SBATCH --output=/home/jaalzate/Prostate_Cancer_TFM/Prostate_Segmentation/Out_files/nnUNet_%j.out

#SBATCH --gres=gpu:1

module load PyTorch
module load GCC

#export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
export PATH="/home/jaalzate/.local/bin:$PATH"
export PYTHONPATH="/home/jaalzate/.local/lib/python3.10/site-packages:/home/jaalzate/BIMCV-AIKit:$PYTHONPATH"

export nnUNet_raw="/nvmescratch/ceib/Prostate/nnUnet/nnUNet_raw"
export nnUNet_preprocessed="/nvmescratch/ceib/Prostate/nnUnet/nnUNet_preprocessed"
export nnUNet_results="/nvmescratch/ceib/Prostate/nnUnet/nnUNet_results"

jupyter lab --ip '0.0.0.0' --port 8888