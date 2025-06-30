EXP_NAME='trial'
IMAGE_NAME='DSC00580'
IMAGE_EXT='jpg'
TRAJ='chair_above'

module load Anaconda2/2019.10-fasrc01 cuda/11.8

source activate /n/home08/adimitriou0/miniconda/envs/cut3r

# Initial CUT3R inference
python demo.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path my_examples/${EXP_NAME} \
    --device cuda \
    --size 512 \
    --vis_threshold 1.5 \
    --output_dir ./output/${EXP_NAME} \
    --exp_name ${EXP_NAME}