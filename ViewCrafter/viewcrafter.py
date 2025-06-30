import sys
sys.path.append('./extern/dust3r')
# sys.path.append('./extern/mast3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
# from mast3r.model import AsymmetricMASt3R
import trimesh
import torch
import numpy as np
import torchvision
import os
import copy
import cv2  
import glob
from PIL import Image
import pytorch3d
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from utils.pvd_utils import *
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis
from pathlib import Path
from torchvision.utils import save_image



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil

class ViewCrafter:
    def __init__(self, opts, gradio=False, step=None):
        """
        Initialize the ViewCrafter pipeline.

        Args:
            opts: Parsed command-line or config options.
            gradio (bool): Flag indicating if the Gradio interface is used.
        
        Loads the DUSt3R stereo model and diffusion model, and initializes the
        input image(s) and 3D scene if Gradio is not being used.
        """
        print("Initializing ViewCrafter...")
        self.opts = opts
        self.device = opts.device
        self.step = step
        self.video_length = 5*step + 5
        print(f"Video Length: {self.video_length}")
        
        if self.opts.pcd_dir is None:
            print("Setting-up DUSt3R...")
            self.setup_dust3r()         # Load depth-based stereo reconstruction model
        # self.setup_mast3r()
        print("Setting-up Diffusion...")
        self.setup_diffusion()      # Load point-conditioned video diffusion model

        # If not in Gradio mode, process the input image(s)
        if not gradio and self.opts.pcd_dir is None:
            if os.path.isfile(self.opts.image_dir):  # Single image input
                print("Run DUSt3R...")
                self.images, self.img_ori = self.load_initial_images(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images)
                # self.run_mast3r(input_images=self.images)
            elif os.path.isdir(self.opts.image_dir):  # Multiple images input
                print("Run DUSt3R...")
                self.images, self.img_ori = self.load_initial_dir(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images, clean_pc = True)
                # self.run_mast3r(input_images=self.images, clean_pc = True)
            else:
                print(f"{self.opts.image_dir} doesn't exist")      

        if self.opts.pcd_dir:
            self.img_ori = torch.from_numpy(np.array(Image.open(self.opts.image_dir).convert("RGB").resize((1024, 576), Image.BILINEAR))).float() / 255.0
            pcd_dict = read_pcs(self.opts.pcd_dir)

            all_pts = []
            all_colors = []

            for step in sorted(pcd_dict.keys()):
                pc = pcd_dict[step]["pc"]
                color = pcd_dict[step]["color"]
                conf = pcd_dict[step]["conf"]
                edge_color = pcd_dict[step].get("edge_color", None)

                pts, color = parse_pc_data(pc, color, conf, edge_color, set_border_color=True)  

                all_pts.append(pts)
                all_colors.append(color)

            self.pcd = np.concatenate(all_pts, axis=0)
            self.colors = np.concatenate(all_colors, axis=0)
        
    def run_dust3r(self, input_images, clean_pc=False):
        """
        Reconstruct a point cloud from one or more input images using the DUSt3R model.

        Args:
            input_images (list): List of preprocessed image tensors or dictionaries.
            clean_pc (bool): Whether to apply cleaning to the raw point cloud output.

        Side Effects:
            Sets `self.scene` with a globally aligned (and optionally cleaned) point cloud.

        Workflow:
            1. Generates image pairs for stereo matching.
            2. Runs DUSt3R to estimate depths and relative poses.
            3. Performs global alignment across all images to build a coherent 3D scene.
            4. Optionally cleans the resulting point cloud.
        """
        # Generate all possible image pairs with symmetry for stereo depth estimation
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)

        # Perform depth estimation and pose prediction using DUSt3R
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        # Run global alignment to fuse depth maps and poses into a global point cloud
        mode = GlobalAlignerMode.PointCloudOptimizer
        scene = global_aligner(output, device=self.device, mode=mode)

        # Optimize alignment across all images using minimum spanning tree initialization
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        # Optionally clean the point cloud to remove noise or outliers
        self.scene = scene.clean_pointcloud() if clean_pc else scene

    def run_mast3r(self, input_images, clean_pc=False):
        """
        Reconstruct a point cloud from one or more input images using the DUSt3R model.

        Args:
            input_images (list): List of preprocessed image tensors or dictionaries.
            clean_pc (bool): Whether to apply cleaning to the raw point cloud output.

        Side Effects:
            Sets `self.scene` with a globally aligned (and optionally cleaned) point cloud.

        Workflow:
            1. Generates image pairs for stereo matching.
            2. Runs DUSt3R to estimate depths and relative poses.
            3. Performs global alignment across all images to build a coherent 3D scene.
            4. Optionally cleans the resulting point cloud.
        """
        # Generate all possible image pairs with symmetry for stereo depth estimation
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)

        # Perform depth estimation and pose prediction using DUSt3R
        output = inference(pairs, self.mast3r, self.device, batch_size=self.opts.batch_size)

        # Run global alignment to fuse depth maps and poses into a global point cloud
        mode = GlobalAlignerMode.PointCloudOptimizer
        scene = global_aligner(output, device=self.device, mode=mode)

        # Optimize alignment across all images using minimum spanning tree initialization
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        # Optionally clean the point cloud to remove noise or outliers
        self.scene = scene.clean_pointcloud() if clean_pc else scene

    def render_pcd(self, pts3d, imgs, masks, views, renderer, device, colors=None):
        """
        Render RGB images from a colored 3D point cloud using the provided renderer.

        Args:
            pts3d (list of np.ndarray): List of 3D point arrays (one per image).
            imgs (list of np.ndarray): Corresponding RGB values for each point.
            masks (list of np.ndarray or None): Binary masks to filter valid points.
            views (int): Number of camera views to render from.
            renderer (PointsRenderer): PyTorch3D point cloud renderer.
            device (torch.device): Target device for tensor operations.
            nbv (bool): If True, also render binary view masks for NBV (next-best-view) search.

        Returns:
            images (tensor): Rendered RGB images.
            view_masks (tensor or None): Rendered masks (only if `nbv` is True).
        """

        if imgs is not None:
            # Convert point and image data from numpy to device tensors
            imgs = to_numpy(imgs)
            pts3d = to_numpy(pts3d)
            # Flatten all points and colors across views
            pts = torch.from_numpy(np.concatenate(pts3d)).view(-1, 3).to(device)
            print(f"Render pcd ({type(pts) = })")
            print(f"Render pcd ({pts.shape = })")
            print(f"Render pcd ({type(pts3d) = })")
            print(f"Render pcd ({pts3d[0].shape = })")
            
            col = torch.from_numpy(np.concatenate(imgs)).view(-1, 3).to(device)
            print(f"Render pcd ({type(col) = })")
            print(f"Render pcd ({col.shape = })")
        else:
            print("Yeeeeeeeeeeeeeeehah")
            pts = torch.from_numpy(pts3d).to(device)
            col = torch.from_numpy(colors).to(device)

        # Construct a point cloud object for rendering across multiple views
        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)

        view_masks = None

        return images, view_masks
    
    def run_render(self, pcd, imgs, masks, H, W, camera_traj, num_views, colors=None):
        """
        Set up the renderer and render novel views from a point cloud along a camera trajectory.

        Args:
            pcd (list): List containing 3D point cloud(s) (as tensors or arrays).
            imgs (list): List of corresponding RGB images.
            masks (list or None): Optional masks to filter valid points.
            H (int): Image height.
            W (int): Image width.
            camera_traj (PerspectiveCameras): Camera path to render from.
            num_views (int): Number of views in the trajectory.
            nbv (bool): Whether to return view masks for NBV selection (currently unused).

        Returns:
            render_results (tensor): Rendered images from point cloud.
            viewmask (tensor or None): View masks (always None here since nbv=False).
        """

        # Initialize renderer using the given trajectory and image size
        render_setup = setup_renderer(camera_traj, image_size=(H, W))
        renderer = render_setup['renderer']

        # Call rendering function on the point cloud
        render_results, viewmask = self.render_pcd(
            pcd, imgs, masks, num_views, renderer, self.device, colors=colors
        )

        return render_results, viewmask
    
    def run_diffusion(self, renderings):
        """
        Generate high-fidelity video frames from coarse point cloud renderings
        using the point-conditioned video diffusion model.

        Args:
            renderings (torch.Tensor): Coarse rendered frames from point cloud, 
                                    shape [T, H, W, 3] in range [0, 1].

        Returns:
            torch.Tensor: Refined video frames, shape [T, H, W, 3], in range [-1, 1].

        Notes:
            - The diffusion model uses both visual features and optional prompts.
            - The input is normalized to [-1, 1] and permuted to [B, C, T, H, W].
            - `image_guided_synthesis` performs iterative denoising based on learned 3D priors.
        """

        prompts = [self.opts.prompt]

        # Normalize input renderings to [-1, 1] and format for the diffusion model
        videos = (renderings * 2. - 1.).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)  # [1, 3, T, H, W]
        condition_index = [0]  # Index of the conditioning frame

        # Inference with automatic mixed precision and no gradients
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_samples = image_guided_synthesis(
                self.diffusion,
                prompts,
                videos,
                self.noise_shape,
                self.opts.n_samples,
                self.opts.ddim_steps,
                self.opts.ddim_eta,
                self.opts.unconditional_guidance_scale,
                self.opts.cfg_img,
                self.opts.frame_stride,
                self.opts.text_input,
                self.opts.multiple_cond_cfg,
                self.opts.timestep_spacing,
                self.opts.guidance_rescale,
                condition_index
            )

        # Post-process output to bring it back to image space
        return torch.clamp(batch_samples[0][0].permute(1, 2, 3, 0), -1., 1.)

    def nvs_single_view(self, gradio=False):
        """
        Perform novel view synthesis from a single image using specified camera trajectory logic.

        Args:
            gradio (bool): Whether Gradio interface is used (affects trajectory source).

        This method:
            - Extracts intrinsics, depth, and point cloud from the reconstructed scene.
            - Computes a spherical camera trajectory based on one of three modes:
                - 'single_view_nbv': selects least-occluded next-best-view (NBV).
                - 'single_view_target': applies fixed angular and positional offsets.
                - 'single_view_txt': loads full phi/theta/r sequence from file or UI.
            - Renders coarse RGB views from point cloud along the trajectory.
            - Optionally interpolates renderings and applies diffusion model to synthesize high-fidelity novel views.
            - Saves outputs (rendered video and optionally candidate masks).
        """
        data = np.load(self.opts.cam_info_dir)
        K = data["intrinsics"]
        pose = data["pose"]

        w2c = np.linalg.inv(pose)
        c2ws = torch.tensor(w2c[None, ...], dtype=torch.float32).to('cuda:0')

        focal = K[0, 0]  # assuming fx = fy
        focals = torch.tensor([[focal]], dtype=torch.float32, device='cuda:0')

        cx, cy = K[0, 2], K[1, 2]
        principal_points = torch.tensor([[cx, cy]], dtype=torch.float32, device='cuda:0')
        

        H, W = 288, 512


        if self.opts.pcd_dir is None:

            # Get camera intrinsics and poses from the point cloud reconstruction
            c2ws = self.scene.get_im_poses().detach()[1:]  # World-to-camera matrices (skip the 0th ref view)
            principal_points = self.scene.get_principal_points().detach()[1:]  # (cx, cy)
            focals = self.scene.get_focals().detach()[1:]  # Focal lengths

            # Extract image resolution from the original input
            shape = self.images[0]['true_shape']
            H, W = int(shape[0][0]), int(shape[0][1])

            # Extract 3D points and depth maps; center depth used to define orbit radius
            self.pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)]
            depth = [i.detach() for i in self.scene.get_depthmaps()]
            depth_avg = depth[-1][H//2,W//2]  # Z-depth at image center
            radius = depth_avg * self.opts.center_scale  # Scaled orbit radius for camera path

            # print(f"{type(pcd) = }")
            # print(f"{pcd[0].shape = }")

            all_points = torch.cat([pts.view(-1, 3) for pts in self.pcd], dim=0)  # shape [N, 3]

            min_vals = all_points.min(dim=0).values
            max_vals = all_points.max(dim=0).values

            depth_save(depth, os.path.join(self.opts.save_dir, f"depth.png"))

            with open(os.path.join(self.opts.save_dir, "image_details.txt"), "w") as f:
                f.write(f"Center depth: {depth_avg}\n")
                f.write(f"Min values (x, y, z): {min_vals.tolist()}\n")
                f.write(f"Max values (x, y, z): {max_vals.tolist()}\n")


            # Convert world coordinates to object-centric frame (used for orbit-style camera movement)
            # c2ws, pcd = world_point_to_obj(
            #     poses=c2ws,
            #     points=torch.stack(self.pcd),
            #     k=-1,  # Selects the last pose as the reference coordinate system
            #     r=radius if self.opts.mode != "single_view_txt_free" else 0,
            #     elevation=self.opts.elevation,
            #     device=self.device
            # )
        
        

            # Prepare image data and (optional) masks
            imgs = np.array(self.scene.imgs)
        masks = None

        if self.opts.mode == 'single_view_target':
            # Direct target offset trajectory based on d_theta/d_phi/d_r and optional XY shifts
            camera_traj, num_views = generate_traj_specified(
                c2ws, H, W, focals, principal_points,
                self.opts.d_theta[0], self.opts.d_phi[0], self.opts.d_r[0],
                self.opts.d_x[0] * depth_avg / focals.item(),
                self.opts.d_y[0] * depth_avg / focals.item(),
                self.opts.video_length, self.device
            )
        elif self.opts.mode == 'single_view_txt':
            print("Single View from text file...")
            # Load custom trajectory from a txt file or UI values
            if not gradio:
                # with open(self.opts.traj_txt, 'r') as file:
                #     lines = file.readlines()
                #     phi = [float(i) for i in lines[0].split()]
                #     theta = [float(i) for i in lines[1].split()]
                #     r = [float(i) for i in lines[2].split()]
                with open(self.opts.traj_txt, 'r') as file:
                    # Read all lines and split each into float values
                    rows = [list(map(float, line.split())) for line in file]
                    # Transpose rows to get columns
                    phi, theta, r = zip(*rows)
            else:
                phi, theta, r = self.gradio_traj

            camera_traj, num_views = generate_traj_txt(
                c2ws, H, W, focals, principal_points,
                phi, theta, r, self.opts.video_length,
                self.device, viz_traj=True, save_dir=os.path.join(self.opts.save_dir, os.path.splitext(os.path.basename(self.opts.traj_txt))[0])
            )

            shutil.copy(self.opts.traj_txt, os.path.join(self.opts.save_dir, f"{os.path.splitext(os.path.basename(self.opts.traj_txt))[0]}.txt"))
        elif self.opts.mode == "single_view_txt_free":
            x = [0.0];  y = [0.0];  z = [0.0]
            yaw = [0.0]; pitch = [0.0]; roll = [0.0]
            with open(self.opts.traj_txt, 'r') as file:
                for dx, dy, dz, dyaw, dpitch, droll in map(lambda l: map(float, l.split()), file):
                    x.append(x[-1] +  dx * self.opts.traj_scale)      # +x
                    y.append(y[-1] +  dy * self.opts.traj_scale)      # −y  (image → world)
                    z.append(z[-1] +  dz * self.opts.traj_scale)      # −z
                    yaw.append(yaw[-1] +  dyaw)         # −yaw
                    pitch.append(pitch[-1] + dpitch)    # +pitch
                    roll.append(roll[-1] +  droll)      # −roll
                print(f"{x = }")
                print(f"{y = }")
                print(f"{z = }")
                print(f"{yaw = }")
                print(f"{pitch = }")
                print(f"{roll = }")

                video_length = 10 * (len(x) - 1)
                print(f"{video_length = }")
            
            camera_traj, num_views, frame_idx = generate_traj_txt_free(
                c2ws, H, W, focals, principal_points,
                x, y, z, yaw, pitch, roll, self.video_length,
                self.device, viz_traj=False, save_dir=os.path.join(self.opts.save_dir, f"{os.path.splitext(os.path.basename(self.opts.traj_txt))[0]}_{self.step}")
            )

            shutil.copyfile(self.opts.traj_txt, os.path.join(os.path.join(self.opts.save_dir, f"{os.path.splitext(os.path.basename(self.opts.traj_txt))[0]}_{self.step}"), 'trajectory.txt'))
        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")
        
        print("Rendering Results...")
        # Render views from the computed camera trajectory
        if self.opts.pcd_dir is None:
            render_results, viewmask = self.run_render(
                [self.pcd[-1]], [imgs[-1]], masks, H, W, camera_traj, num_views
            )
            # save_video(render_results, os.path.join(os.path.join(self.opts.save_dir, f"{os.path.splitext(os.path.basename(self.opts.traj_txt))[0]}_{self.step}"), 'render1.mp4'))
        else:
            render_results, viewmask = self.run_render(
                self.pcd, None, masks, 576, 1024, camera_traj, num_views, colors=self.colors
            )
            # save_video(render_results, os.path.join(os.path.join(self.opts.save_dir, f"{os.path.splitext(os.path.basename(self.opts.traj_txt))[0]}_{self.step}"), 'render2.mp4'))


        # Resize renderings to match expected diffusion model input (e.g., 576x1024)
        render_results = F.interpolate(
            render_results.permute(0, 3, 1, 2),  # NHWC → NCHW
            size=(576, 1024),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # NCHW → NHWC

        # Set the first frame to be the original input image
        for i in range(len(frame_idx) - 1): # Skip the last one (we haven't calculated it yet)
            print(f"{frame_idx[i] = }")
            frame_dir = f'../results/{self.opts.exp_id}/step{str(i).zfill(2)}/rgb.png'
            render_results[frame_idx[i]] = torch.from_numpy(np.array(Image.open(frame_dir).convert("RGB").resize((1024, 576), Image.BILINEAR))).float() / 255.0
            pass
        render_results[0] = self.img_ori

        # Handle edge case: if trajectory ends in (0,0,0), loop the video by copying the first frame to last
        # if self.opts.mode == 'single_view_txt':
        #     if phi[-1] == 0. and theta[-1] == 0. and r[-1] == 0.:
        #         render_results[-1] = self.img_ori

        # if self.opts.mode == 'single_view_txt_free':
        #     if x[-1] == 0. and y[-1] == 0. and z[-1] == 0. and yaw[-1] == 0. and pitch[-1] == 0. and roll[-1] == 0.:
        #         render_results[-1] = self.img_ori

        # Save rendered trajectory as video
        save_video(render_results, os.path.join(os.path.join(self.opts.save_dir, f"{os.path.splitext(os.path.basename(self.opts.traj_txt))[0]}_{self.step}"), 'render.mp4'))

        # Optionally save point cloud with camera poses for visualization
        if self.opts.pcd_dir is None:
            save_pointcloud_with_normals([imgs[-1]], [self.pcd[-1]], msk=None, save_path=os.path.join(self.opts.save_dir,'pcd.ply') , mask_pc=False, reduce_pc=False)

        # Refine coarse point cloud renderings using the diffusion model
        diffusion_results = self.run_diffusion(render_results)

        # Save diffusion output as a video (normalized back to [0, 1])
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(os.path.join(self.opts.save_dir, f"{os.path.splitext(os.path.basename(self.opts.traj_txt))[0]}_{self.step}"), 'diffusion.mp4'))

        save_path  = os.path.join(os.path.join(self.opts.save_dir, f"{os.path.splitext(os.path.basename(self.opts.traj_txt))[0]}_{self.step}"), 'last_frame.jpg')

        os.makedirs(f"../results/{self.opts.exp_id}/step{str(self.step).zfill(2)}", exist_ok=True)

        img = diffusion_results[-1].detach().cpu()          # H×W×3, float16/32, range [-1,1]
        img = ((img + 1) / 2).float()                       # → float32, [0,1]
        img = (img * 255).to(torch.uint8).numpy()           # → uint8, [0,255]

        if not self.opts.testing:
            Image.fromarray(img).save(save_path, quality=95)
            Image.fromarray(img).save(f'../CUT3R/my_examples/{self.opts.exp_id}/frame_{str(self.step).zfill(3)}.jpg', quality=95)
            Image.fromarray(img).save(f'../results/{self.opts.exp_id}/step{str(self.step).zfill(2)}/rgb.png', quality=95)

        print("Execution Over.")

        return diffusion_results

    def setup_diffusion(self):
        seed_everything(self.opts.seed)

        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())

        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

        h, w = self.opts.height // 8, self.opts.width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = self.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def setup_dust3r(self):
        self.dust3r = load_model(self.opts.model_path, self.device)

    def setup_mast3r(self):
        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        self.mast3r = AsymmetricMASt3R.from_pretrained(model_name).to(self.device)
    
    def load_initial_images(self, image_dir):
        """
        Load and preprocess the initial image(s) for depth estimation and point cloud reconstruction.

        Parameters:
            image_dir (str): Path to the input image file.

        Returns:
            Tuple[List[Dict], np.ndarray]:
                - images: A list of dictionaries containing the loaded and preprocessed image tensors.
                        Each dict contains keys: 'img', 'true_shape', 'idx', 'instance', 'img_ori'.
                        These are required by the DUSt3R depth estimation pipeline.
                - img_ori: Original RGB image in numpy format, range [0, 1], shape (H, W, 3).

        Notes:
            - The image is resized to 512 during loading via DUSt3R's `load_images()` utility.
            - If only a single image is provided, it is duplicated to form a pair, because stereo-based
            pipelines require at least two images.
        """

        # Load the image as a dictionary containing all necessary meta-information.
        # `force_1024=True` enforces a standard output shape of (576, 1024) during resizing.
        images = load_images([image_dir], size=512, force_1024=True)

        # Extract the original image (after resizing) and convert to numpy format with range [0, 1]
        img_ori = (images[0]['img_ori'].squeeze(0).permute(1, 2, 0) + 1.0) / 2.0  # (H, W, 3), float32

        # If only one image is loaded, duplicate it to satisfy stereo input requirements
        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1  # Differentiate the second image by assigning a new index

        return images, img_ori

    def load_initial_dir(self, image_dir):

        image_files = glob.glob(os.path.join(image_dir, "*"))

        if len(image_files) < 2:
            raise ValueError("Input views should not less than 2.")
        image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        images = load_images(image_files, size=512,force_1024 = True)

        img_gts = []
        for i in range(len(image_files)):
            img_gts.append((images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.) 

        return images, img_gts

    def run_gradio(self,i2v_input_image, i2v_elevation, i2v_center_scale, i2v_d_phi, i2v_d_theta, i2v_d_r, i2v_steps, i2v_seed):
        self.opts.elevation = float(i2v_elevation)
        self.opts.center_scale = float(i2v_center_scale)
        self.opts.ddim_steps = i2v_steps
        self.gradio_traj = [float(i) for i in i2v_d_phi.split()],[float(i) for i in i2v_d_theta.split()],[float(i) for i in i2v_d_r.split()]
        seed_everything(i2v_seed)
        torch.cuda.empty_cache()
        img_tensor = torch.from_numpy(i2v_input_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        img_tensor = (img_tensor / 255. - 0.5) * 2

        image_tensor_resized = center_crop_image(img_tensor) #1,3,h,w
        images = get_input_dict(image_tensor_resized,idx = 0,dtype = torch.float32)
        images = [images, copy.deepcopy(images)]
        images[1]['idx'] = 1
        self.images = images
        self.img_ori = (image_tensor_resized.squeeze(0).permute(1,2,0) + 1.)/2.

        # self.images: torch.Size([1, 3, 288, 512]), [-1,1]
        # self.img_ori:  torch.Size([576, 1024, 3]), [0,1]
        # self.images, self.img_ori = self.load_initial_images(image_dir=i2v_input_image)
        self.run_dust3r(input_images=self.images)
        self.nvs_single_view(gradio=True)

        traj_dir = os.path.join(self.opts.save_dir, "viz_traj.mp4")
        gen_dir = os.path.join(self.opts.save_dir, "diffusion0.mp4")
        
        return traj_dir, gen_dir



    def visualize_depth_maps(self, depth_list, titles=None, cmap='plasma'):
        n = len(depth_list)
        plt.figure(figsize=(12, 5))

        for i, d in enumerate(depth_list):
            d_np = d.detach().cpu().numpy()

            plt.subplot(1, n, i + 1)
            plt.imshow(d_np, cmap=cmap)
            plt.colorbar(label='Depth')
            if titles:
                plt.title(titles[i])
            else:
                plt.title(f"Depth Map {i}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"depth_{os.path.splitext(os.path.basename(self.opts.image_dir))[0]}.png")

        