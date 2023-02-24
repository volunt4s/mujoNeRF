from src.generate.load_nerf_data import load_mujoco_data

import src.nerf.helper as hp 
import numpy as np
import torch
import matplotlib.pyplot as plt


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

if __name__ == "__main__":
    images, poses, [H, W, focal], i_split = load_mujoco_data()

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    ################
    # Hyperparameter
    near = 1.0
    far = 4.0
    N_rand = 1024
    N_samples = 64

    args = {
        "multires" : 10,
        "i_embed" : 0,
        "use_viewdirs" : True,
        "multires_views" : 4,
        "N_importance" : 128,
        "netdepth" : 8,
        "netwidth" : 128,
        "netdepth_fine" : 8,
        "netwidth_fine" : 128,
        "netchunk" : 65536,
        "lrate" : 0.0005,
        "perturb" : 1.0,
        "N_samples" : 64,
        "white_bkgd" : False,
        "raw_noise_std" : 0.0
    }
    ##################
    
    img_i = 0
    pose = poses[img_i, :3, :4]
    target = images[img_i]
    target = torch.Tensor(target).to(device)

    render_kwargs_train, start, grad_vars, optimizer = hp.create_nerf(args)
    bds_dict = {
        'near' : near,
        'far' : far
    }
    render_kwargs_train.update(bds_dict)

    # 이미지로부터 ray 받아오기
    rays_o, rays_d = hp.get_rays(H, W, K, torch.Tensor(pose, device=device))
    
    # ray batch 샘플링
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)
    coords = torch.reshape(coords, [-1, 2])
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
    select_coords = coords[select_inds].long()
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
    batch_rays = torch.stack([rays_o, rays_d], 0)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]

    # render
    hp.render(H, W, K, rays=batch_rays, retraw=True, **render_kwargs_train)