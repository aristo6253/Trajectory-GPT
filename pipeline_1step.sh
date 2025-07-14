#!/bin/bash
set -e  # exit on any error

EXP_NAME='testT'
IMAGE_NAME='outdoors2'
IMAGE_EXT='jpg'
TRAJ='black_carT'

# --pcd_dir ../results/${EXP_NAME}/step$(printf "%02d" $((i - 1)))/pcs_data.json \

echo "EXP_NAME: ${EXP_NAME}"
echo "IMAGE_NAME: ${IMAGE_NAME}"
echo "IMAGE_EXT: ${IMAGE_EXT}"
echo "TRAJ: ${TRAJ}"
# echo "TRAJ_DESCRIPTION: ${TRAJ_DESCRIPTION}"

# Initial setup
echo SETTING UP EXPERIMENT
mkdir -p CUT3R/my_examples/${EXP_NAME}
mkdir -p results/${EXP_NAME}/step00
# touch results/${EXP_NAME}/${TRAJ}.txt
cp images/${IMAGE_NAME}.${IMAGE_EXT} CUT3R/my_examples/${EXP_NAME}/frame_000.png
cp images/${IMAGE_NAME}.${IMAGE_EXT} results/${EXP_NAME}/step00/rgb.png

cd CUT3R

module load Anaconda2/2019.10-fasrc01 cuda/11.8
source activate /n/home08/adimitriou0/miniconda/envs/cut3r

echo RUNNING CUT3R

# Initial CUT3R inference
python demo.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path my_examples/${EXP_NAME} \
    --device cuda \
    --size 512 \
    --vis_threshold 1.5 \
    --output_dir ./output/${EXP_NAME} \
    --exp_name ${EXP_NAME}

cd ../

conda deactivate

cd ViewCrafter
echo "LOADING MODULES"

echo "ACTIVATING ENVIRONMENT"
source activate /n/home08/adimitriou0/miniconda/envs/viewcrafter

echo "RUNNING VIEWCRAFTER"
python inference.py \
    --image_dir ../images/${IMAGE_NAME}.${IMAGE_EXT} \
    --out_dir ./output \
    --traj_txt ../results/${EXP_NAME}/${TRAJ}.txt \
    --mode single_view_txt_free \
    --exp_id ${EXP_NAME} \
    --seed 123 \
    --ckpt_path ./checkpoints/model.ckpt \
    --config configs/inference_pvd_1024.yaml \
    --ddim_steps 50 \
    --device cuda:0 \
    --height 576 \
    --width 1024 \
    --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    --prompt "Smooth navigation through a scene" \
    --traj_scale 1 \
    --cam_info_dir ../results/${EXP_NAME}/camera.npz

conda deactivate

cd ../