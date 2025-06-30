from viewcrafter import ViewCrafter
import os
from configs.infer_config import get_parser
from utils.pvd_utils import *
from datetime import datetime
import shutil

if __name__ == "__main__":
    print("Execution Started.")
    # Parse all command-line arguments
    parser = get_parser()
    opts = parser.parse_args()

    # If no experiment name is provided, generate one from timestamp + image file name
    if opts.exp_name is None:
        opts.exp_name = f'{os.path.splitext(os.path.basename(opts.image_dir))[0]}'

    # Create directory to save outputs for this run
    opts.save_dir = os.path.join(opts.out_dir, opts.exp_name) if not opts.testing else 'test_output'
    os.makedirs(opts.save_dir, exist_ok=True)
    with open(opts.traj_txt, 'r') as f:
        step = sum(1 for _ in f)
    os.makedirs(os.path.join(opts.save_dir, f"{os.path.splitext(os.path.basename(opts.traj_txt))[0]}_{step}"), exist_ok=True)

    ext = os.path.splitext(opts.image_dir)[1]
    shutil.copyfile(opts.image_dir, os.path.join(opts.save_dir, f"{os.path.splitext(os.path.basename(opts.image_dir))[0]}{ext}"))

    # Instantiate the ViewCrafter pipeline with the user-defined options
    pvd = ViewCrafter(opts, step=step)

    # Dispatch to the appropriate mode
    if opts.mode == 'single_view_target':
        pvd.nvs_single_view()  # Run synthesis to target pose

    elif opts.mode == 'single_view_txt':
        pvd.nvs_single_view()  # Run synthesis from .txt trajectory

    elif opts.mode == 'single_view_txt_free':
        pvd.nvs_single_view()  # Run synthesis from .txt trajectory

    elif opts.mode == 'single_view_eval':
        pvd.nvs_single_view_eval()  # Presumably for benchmarking or metric evaluation

    elif opts.mode == 'sparse_view_interp':
        pvd.nvs_sparse_view_interp()  # Render interpolated views from sparse input

    else:
        raise KeyError(f"Invalid Mode: {opts.mode}")  # Catch misconfigured mode