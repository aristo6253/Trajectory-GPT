import importlib
import numpy as np
import cv2
import torch
import torch.distributed as dist
from collections import OrderedDict
import os
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from einops import rearrange, repeat

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data   


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z

def image_guided_synthesis(
    model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.,
    unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False,
    multiple_cond_cfg=False, timestep_spacing='uniform', guidance_rescale=0.0,
    condition_index=None, **kwargs
):
    """
    Generate image or video sequences conditioned on both image features and optional text prompts 
    using a DDIM sampler.

    Args:
        model: The diffusion model with conditioning and decoding capabilities.
        prompts (List[str]): List of text prompts, one per batch element.
        videos (Tensor): Input video tensor of shape (B, C, T, H, W), used for conditioning.
        noise_shape (Tuple[int]): Shape of the latent noise input (B, C, T, H, W).
        n_samples (int): Number of samples to generate per input in the batch.
        ddim_steps (int): Number of DDIM sampling steps.
        ddim_eta (float): Controls stochasticity of DDIM (0 for deterministic).
        unconditional_guidance_scale (float): CFG scale for combining conditional and unconditional paths.
        cfg_img (float or None): Guidance scale specifically for image conditioning (if multiple_cond_cfg=True).
        fs (int or List[int]): Optional frame-step conditioning input per sample.
        text_input (bool): If False, disables text conditioning and uses empty prompts.
        multiple_cond_cfg (bool): If True, enables separate unconditional conditioning path for image-only input.
        timestep_spacing (str): Sampling strategy for timesteps (e.g., "uniform", "linear").
        guidance_rescale (float): Rescale factor for classifier-free guidance correction.
        condition_index (List[int]): Time indices from `videos` to use for image conditioning.
        **kwargs: Additional parameters passed to the DDIM sampler.

    Returns:
        Tensor: Synthesized batch of shape (B, n_samples, C, T, H, W).
    """
    # Initialize the sampler based on whether multiple conditioning configs are used
    ddim_sampler = DDIMSampler_multicond(model) if multiple_cond_cfg else DDIMSampler(model)
    
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    # Use empty prompts if not using text input
    if not text_input:
        prompts = [""] * batch_size
    assert condition_index is not None, "Error: condition index is None!"

    # Extract and project image features
    img = videos[:, :, condition_index[0]]  # (B, C, H, W)
    img_emb = model.embedder(img)           # (B, L, C)
    img_emb = model.image_proj_model(img_emb)

    # Get text conditioning and combine with image embedding
    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}

    # Add additional image feature map if using hybrid conditioning
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos)  # (B, C, T, H, W)
        cond["c_concat"] = [z]

    # Prepare unconditional conditioning if guidance is used
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            uc_emb = model.get_learned_conditioning([""] * batch_size)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)

        uc_img_emb = model.embedder(torch.zeros_like(img))
        uc_img_emb = model.image_proj_model(uc_img_emb)

        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [z]
    else:
        uc = None

    # Optional: second unconditional config (textless, image-conditioned)
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [z]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None
    batch_variants = []

    # Generate samples
    for _ in range(n_samples):
        cond_z0 = z0.clone() if z0 is not None else None
        if cond_z0 is not None:
            kwargs.update({"clean_cond": True})

        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            cfg_img=cfg_img,
            mask=cond_mask,
            x0=cond_z0,
            fs=fs,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            **kwargs
        )

        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)

    # Output shape: (B, n_samples, C, T, H, W)
    return torch.stack(batch_variants).permute(1, 0, 2, 3, 4, 5)
