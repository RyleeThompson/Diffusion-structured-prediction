#!/bin/bash

#SBATCH --job-name=EMA-test-50-steps
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=t4v1,t4v2,p100,t4v2,rtx6000
#SBATCH --ntasks=1
#SBATCH --time=4-0:0:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rylee@uoguelph.ca
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err

# module load singularity-ce
singularity run --nv --bind /checkpoint/rylee,$COCOpath diff.sif python train.py model/backbone=gt model.backbone.static_noise=True model.backbone.num_timesteps=4000 model.backbone.timestep=0.025 model/structured/conditioning=bb_preds model.structured.conditioning.concat_method=seq diffusion.num_timesteps=50 +ckpt_path=/checkpoint/rylee/$SLURM_JOB_ID
rsync -r /checkpoint/rylee/$SLURM_JOB_ID ~/Diffusion-structured-prediction
