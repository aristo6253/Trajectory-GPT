#!/bin/bash
set -e  # exit on any error

EXP_NAME='align_r2412_nc'
IMAGE_NAME='room2412'
IMAGE_EXT='jpg'
# TRAJ_DESCRIPTION="Go to the black car visible on this scene, by first yawing right until aligned and then moving forward until you reach the car."
# TRAJ_DESCRIPTION="Reach the chair in the corner of the room by flying over the table making sure not to collide with the table or the chairs in the middle of the scene. You should keep the target chair always in frame and stop when the target chair is the center of the view."
# TRAJ_DESCRIPTION="Reach the blue trash bin in the end of the room by adjusting the yaw and pitch to have the target centered and after the centering is done advance towards the objective until you are in front of it."
# TRAJ_DESCRIPTION="Move towards the red couch in front of you and then turn right in order to get to the open space" 
TRAJ_DESCRIPTION="Yaw and pitch until you are perfectly aligned with the blue trash can. Meaning that the trash can is in the center of the image"

# --prompt "${TRAJ_DESCRIPTION}" \
# --prompt "Navigation through an indoor scene" \
DIFF_PROMPT="Smooth navigation through a scene"


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
cp images/${IMAGE_NAME}.${IMAGE_EXT} CUT3R/my_examples/${EXP_NAME}/frame_000.png
cp images/${IMAGE_NAME}.${IMAGE_EXT} results_diff/${EXP_NAME}/step00/rgb.png

# Start loop
for i in $(seq 1 20); do
    echo "========== ITERATION $i =========="

    cd CUT3R

    module load Anaconda2/2019.10-fasrc01 cuda/11.8
    source activate /n/home08/adimitriou0/miniconda/envs/cut3r

    echo RUNNING CUT3R

    if [ "$i" -eq 1 ]; then
        script="demo.py"
    else
        script="demo_ga.py"
    fi

    # Initial CUT3R inference
    python "$script" \
        --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path my_examples/${EXP_NAME} \
        --device cuda \
        --size 512 \
        --vis_threshold 1.5 \
        --output_dir ./output/${EXP_NAME} \
        --exp_name ${EXP_NAME}

    cd ../

    # Initial GPT prompt
    python gpt_prompter_diff.py \
        --traj_desc "${TRAJ_DESCRIPTION}" \
        --exp_name ${EXP_NAME} \
        --incr_file results_diff/${EXP_NAME}/increments.txt \
        --logic_file results_diff/${EXP_NAME}/logic.txt \
        # --preplanned_traj
        # --overlay_cross 

    conda deactivate

    cd ViewCrafter
    echo "LOADING MODULES"

    echo "ACTIVATING ENVIRONMENT"
    source activate /n/home08/adimitriou0/miniconda/envs/viewcrafter

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
        --pcd_dir ../results_diff/${EXP_NAME}/step$(printf "%02d" $((i - 1)))/pcs_data.json \
        --traj_scale 1 \
        --cam_info_dir ../results_diff/${EXP_NAME}/camera.npz

    conda deactivate

    cd ../

done

echo "===== DONE WITH ALL ITERATIONS ====="
