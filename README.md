# Trajectory-GPT

## Setup Diffusion Pipeline
We need to setup both CUT3R and Viewcrafter:

### Viewcrafter

```bash
# Create conda environment
conda create -n viewcrafter python=3.9.16
conda activate viewcrafter
pip install -r requirements_viewcrafter.txt

# Install PyTorch3D
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu117_pyt1131.tar.bz2

# Download pretrained DUSt3R model
cd ViewCrafter
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
cd ../
conda deactivate
```

### Download Checkpoints

We have already downloaded the DUSt3R checkpoints, now we need to dowload Viewcrafter's LDM weights and add them in the `./checkpoints` directory.

|Model|Resolution|Frames|GPU Mem. & Inference Time (tested on a 40G A100, ddim 50 steps)|Checkpoint|Description|
|:---------|:---------|:--------|:--------|:--------|:--------|
|ViewCrafter_25|576x1024|25| 23.5GB & 120s (`perframe_ae=True`)|[Hugging Face](https://huggingface.co/Drexubery/ViewCrafter_25/blob/main/model.ckpt)|Used for single view NVS, can also adapt to sparse view NVS|

### CUT3R

```bash
cd CUT3R
conda create -n cut3r python=3.11 cmake=3.14.0
conda activate cut3r
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install -r requirements_cut3r.txt
conda install 'llvm-openmp<16'
# for training logging
pip install git+https://github.com/nerfstudio-project/gsplat.git # Do we need?

cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```
### Download Checkpoints

We currently provide checkpoints on Google Drive:

| Modelname   | Training resolutions | #Views| Head |
|-------------|----------------------|-------|------|
| [`cut3r_224_linear_4.pth`](https://drive.google.com/file/d/11dAgFkWHpaOHsR6iuitlB_v4NFFBrWjy/view?usp=drive_link) | 224x224 | 16 | Linear |
| [`cut3r_512_dpt_4_64.pth`](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link) | 512x384, 512x336, 512x288, 512x256, 512x160, 384x512, 336x512, 288x512, 256x512, 160x512 | 4-64 | DPT |

> `cut3r_224_linear_4.pth` is our intermediate checkpoint and `cut3r_512_dpt_4_64.pth` is our final checkpoint.

To download the weights, run the following commands:
```bash
cd src
# for 224 linear ckpt
gdown --fuzzy https://drive.google.com/file/d/11dAgFkWHpaOHsR6iuitlB_v4NFFBrWjy/view?usp=drive_link 
# for 512 dpt ckpt
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ../..
```

## Setup 3D GS Pipeline

```bash
cd gaussian-splatting
# Create environment
conda create -n gaussian_splatting python=3.10 -y
conda activate gaussian_splatting

# Install PyTorch 2.0.1 with CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118

# Other dependencies
pip install plyfile tqdm opencv-python joblib

# Install submodules
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/fused-ssim

```


Need COLMAP setup for new scenes to be included, when it is, we need to have the collection of our images under `./scene_dir/input` and run:
```bash
python convert.py -s <scene_dir> --no_gpu
```
When the COLMAP of the scene is created we can train the gaussian-splatting:
```bash
python train.py -s <scene_dir>
```


### Run Diffusion Pipeline
To run the diffusion pipeline we will be running the `pipeline_diff.sh` script. In this we should define the following variables: `EXP_NAME`, `IMAGE_NAME`,`IMAGE_EXT` and `TRAJ_DESCRIPTION`.

The resulting steps (inputs for gpt) will show-up under the directory `results_diff`.

### Run 3D GS Pipeline
To run the 3D GS pipeline we will be running the `pipeline_3dgs.sh` script. In this we should define the following variables: `EXP_NAME`, `TRAJ_DESCRIPTION` and `MODEL`. Within the model directory you can add the camera info of the first camera pose as json file called `source_trajectory.json` with the structure of the following example:
```json
[
    {
        "id": 0,
        "img_name": "step00/rgb.png",
        "width": 1293,
        "height": 837,
        "position": [
            0.40299373818649276,
            2.856104028906749,
            -3.319456188408219
        ],
        "rotation": [
            [
                0.9990924349623449,
                0.04148203019771042,
                -0.009671999363558828
            ],
            [
                -0.04258370998266792,
                0.9675917831662576,
                -0.24890313134481437
            ],
            [
                -0.0009664600997814753,
                0.24908910518089333,
                0.9684800894365652
            ]
        ],
        "fy": 965.517285933431,
        "fx": 965.0432236551961
    }
]
```

The resulting steps (inputs for gpt) will show-up under the directory `results_3dgs`.