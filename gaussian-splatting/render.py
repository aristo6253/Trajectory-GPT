#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name))
        torchvision.utils.save_image(gt, os.path.join(gts_path,  view.image_name))

def render_traj(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, trajectory_file):
    traj_path = os.path.join(model_path, name, "ours_{}".format(iteration), os.path.splitext(os.path.basename(trajectory_file))[0])

    makedirs(traj_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]

        # torchvision.utils.save_image(rendering, os.path.join(traj_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(traj_path, view.image_name))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, my_traj: bool, trajectory_file: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, is_traj=my_traj, trajectory_file=trajectory_file)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # print(f"Traj Cams")
        # for i, cam in enumerate(scene.getTrajCameras()):
        #     if cam.image_name == "DSC07964.png":
        #         print(f"{i = }")
        #         print(f"{cam.image_name = }")
        #         print(f"{cam.R = }")
        #         print(f"{cam.T = }")
        #         print(f"{cam.camera_center = }")
        #         print(f"{cam.world_view_transform = }")
        #         print(f"{cam.FoVx = :.2f}, {cam.FoVy = :.2f}")

        # print(f"Train Cams")
        # for i, cam in enumerate(scene.getTrainCameras()):
        #     if cam.image_name == "DSC07964.png":
        #         print(f"{i = }")
        #         print(f"{cam.image_name = }")
        #         print(f"{cam.R = }")
        #         print(f"{cam.T = }")
        #         print(f"{cam.camera_center = }")
        #         print(f"{cam.world_view_transform = }")
        #         print(f"{cam.FoVx = :.2f}, {cam.FoVy = :.2f}")

        if not skip_train:
             print("Train")
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             print("Test")
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if my_traj:
            print("Trajectory")
            render_traj(dataset.model_path, "traj", scene.loaded_iter, scene.getTrajCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, trajectory_file)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--my_traj', action='store_true')
    parser.add_argument('--trajectory_file', type=str, default=None)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, my_traj=args.my_traj, trajectory_file=args.trajectory_file) # Add args