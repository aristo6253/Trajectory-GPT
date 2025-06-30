#!/bin/bash
#SBATCH --job-name=viewcrafter_gpu
#SBATCH --output=logs/viewcrafter_last.out
#SBATCH --error=logs/viewcrafter_last.err
#SBATCH --partition=gpu         # or gpu_test if testing
#SBATCH --gres=gpu:1            # request 1 GPU
#SBATCH --cpus-per-task=4       # CPU cores
#SBATCH --mem=32G               # RAM
#SBATCH --time=01:00:00         # Max runtime (HH:MM:SS)

# Load necessary modules
module load Anaconda2/2019.10-fasrc01 cuda/11.8  # Adjust CUDA version if needed

# Activate your conda environment
source activate /n/home08/adimitriou0/miniconda/envs/viewcrafter

# Run your script
python inference.py \
--image_dir test/images/DSC01488.JPG \
--out_dir ./output \
--traj_txt test/trajs/test.txt \
--mode single_view_txt_free \
--center_scale 1. \
--elevation 5 \
--seed 123 \
--ckpt_path ./checkpoints/model.ckpt \
--config configs/inference_pvd_1024.yaml \
--ddim_steps 50 \
--video_length 10 \
--device cuda:0 \
--height 576 \
--width 1024 \
--model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
