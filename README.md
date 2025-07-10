# Trajectory-GPT

## Setup
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

### CUT3R

```bash
conda create -n cut3r python=3.11 cmake=3.14.0
conda activate cut3r
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install -r requirements_cut3r.txt
conda install 'llvm-openmp<16'
# for training logging
pip install git+https://github.com/nerfstudio-project/gsplat.git # Do we need?
```