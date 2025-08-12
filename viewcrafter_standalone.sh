#!/bin/bash
set -e  # exit on any error

EXP_NAME='base_vc_spin'
IMAGE_NAME='truck'
IMAGE_EXT='jpg'

# TRAJ_DESCRIPTION="Move towards the black door by avoiding the table in front of you, not going over it but sliding to the left and then moving towards our objective, the black door."
TRAJ_DESCRIPTION="Move towards the yellow umbrella by avoiding the truck in front of you, sliding to the left and then moving towards our objective."
# TRAJ_DESCRIPTION="Move towards the black vase by avoiding the table in front of you, not going over it but sliding to the right and then moving towards our objective, the black vase."
# TRAJ_DESCRIPTION="Move towards the yellow excavator by yawing until aligned with it and move forward."

# --prompt "${TRAJ_DESCRIPTION}" \
# --prompt "Navigation through an indoor scene" \
DIFF_PROMPT="Nonesense"


echo "EXP_NAME: ${EXP_NAME}"
echo "IMAGE_NAME: ${IMAGE_NAME}"
echo "IMAGE_EXT: ${IMAGE_EXT}"
# echo "TRAJ: ${TRAJ}"
echo "TRAJ_DESCRIPTION: ${TRAJ_DESCRIPTION}"

# Initial setup
echo SETTING UP EXPERIMENT
mkdir -p CUT3R/my_examples/${EXP_NAME}
mkdir -p results_diff/${EXP_NAME}/step00
touch results_diff/${EXP_NAME}/increments.txt
touch results_diff/${EXP_NAME}/logic.txt
cp images/${IMAGE_NAME}.${IMAGE_EXT} CUT3R/my_examples/${EXP_NAME}/frame_000.jpg
cp images/${IMAGE_NAME}.${IMAGE_EXT} results_diff/${EXP_NAME}/step00/rgb.png

source activate /n/home08/adimitriou0/miniconda/envs/viewcrafter

# Start loop
for i in $(seq 1 20); do
    echo "========== ITERATION $i =========="

    # Initial GPT prompt
    python gpt_prompter_diff.py \
        --traj_desc "${TRAJ_DESCRIPTION}" \
        --exp_name ${EXP_NAME} \
        --incr_file results_diff/${EXP_NAME}/increments.txt \
        --logic_file results_diff/${EXP_NAME}/logic.txt \
        --overlay_cross \
        --preplanned_traj results_diff/train.txt

    mkdir -p results_diff/${EXP_NAME}/step$(printf "%02d" $((i)))/

    cd ViewCrafter

    echo "RUNNING VIEWCRAFTER"
    python inference.py \
        --image_dir ../images/${IMAGE_NAME}.${IMAGE_EXT} \
        --out_dir ./output \
        --traj_txt ../results_diff/${EXP_NAME}/increments.txt \
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
        --prompt "${DIFF_PROMPT}" \
        --traj_scale 0.1 \

    cd ../

done

echo "===== DONE WITH ALL ITERATIONS ====="
