#!/bin/bash
set -e  # exit on any error

EXP_NAME='garden_align_nc002'
# TRAJ_DESCRIPTION="Move towards the black door by avoiding the table in front of you, not going over it but sliding to the left and then moving towards our objective, the black door."
# TRAJ_DESCRIPTION="Can you go around the table making walking in a square avoiding going over the table start by turning left and then start your path" 
TRAJ_DESCRIPTION="Yaw and pitch so that the black door to the top-left is aligned perfectly with the camera."
# TRAJ_DESCRIPTION="Yaw to the left and have a rightwards motion to explore the scene by moving around the table."
MODEL='output/garden_test'
SOURCE_TRAJ="/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aristo/TrajectoryGPT/gaussian-splatting/output/garden_test/source_trajectory.json"

echo "========== EXPERIMENT SETUP =========="
echo "EXP_NAME: ${EXP_NAME}"
echo "MODEL: ${MODEL}"
echo "TRAJ_DESCRIPTION: ${TRAJ_DESCRIPTION}"

mkdir -p results_3dgs/${EXP_NAME}/step00
cp ${SOURCE_TRAJ} results_3dgs/${EXP_NAME}/trajectory.json
cp results_3dgs/${EXP_NAME}/trajectory.json gaussian-splatting/${MODEL}/${EXP_NAME}.json
touch results_3dgs/${EXP_NAME}/increments.txt
touch results_3dgs/${EXP_NAME}/logic.txt

echo "ACTIVATING ENVIRONMENT: gaussian_splatting"
source activate gaussian_splatting

for i in $(seq 0 19); do
    STEP_DIR=results_3dgs/${EXP_NAME}/step$(printf "%02d" $i)
    echo "========== ITERATION $i =========="
    echo "Creating step directory: ${STEP_DIR}"
    mkdir -p ${STEP_DIR}

    echo "RENDERING VIEW FROM TRAJECTORY JSON"
    cd gaussian-splatting
    python render.py -m ${MODEL} \
     --my_traj \
     --trajectory_file ${EXP_NAME}.json \
     --exp_name ${EXP_NAME}

    cd ..

    echo "COMPUTING BEV DEPTH"
    python depth_bev.py \
        --ply gaussian-splatting/${MODEL}/point_cloud/iteration_30000/point_cloud.ply \
        --json gaussian-splatting/${MODEL}/${EXP_NAME}.json \
        --step ${i} \
        --exp_name ${EXP_NAME}

    echo "GENERATING GPT TRAJECTORY STEP"
    python gpt_prompter_3dgs.py \
        --traj_desc "${TRAJ_DESCRIPTION}" \
        --exp_name ${EXP_NAME} \
        --traj_json results_3dgs/${EXP_NAME}/trajectory.json \
        --incr_file results_3dgs/${EXP_NAME}/increments.txt \
        --logic_file results_3dgs/${EXP_NAME}/logic.txt \
        --model gaussian-splatting/${MODEL} \
        # --overlay_cross \
        # --preplanned_traj results_3dgs/explore.txt

    # Add here any post-processing step to turn GPT output into extrinsics if needed

    echo "========== END OF ITERATION $i =========="
done

echo "========== ALL ITERATIONS COMPLETE =========="
