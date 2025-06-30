module load Anaconda2/2019.10-fasrc01 cuda/11.8

source activate /n/home08/adimitriou0/miniconda/envs/viewcrafter

python inference.py \
--image_dir my_images/DSC00580.jpg \
--out_dir ./output \
--traj_txt my_trajs/chair_above.txt \
--mode single_view_txt_free \
--exp_id '004' \
--seed 123 \
--ckpt_path ./checkpoints/model.ckpt \
--config configs/inference_pvd_1024.yaml \
--ddim_steps 50 \
--video_length 60 \
--device cuda:0 \
--height 576 \
--width 1024 \
--model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
--prompt "Navigation through an indoor scene" \
--testing True \
# --pcd_dir ../pcs_data.json