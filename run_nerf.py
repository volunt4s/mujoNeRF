from load_nerf_data import load_data
import nerf_helpers as hp
import numpy as np
import torch
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    images, poses, [H, W, focal] = load_data()

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    ################
    # Hyperparameter
    near = 2.
    far = 6.
    N_rand = 1024
    N_samples = 64
    ##################


    img_i = 0
    pose = poses[img_i, :3, :4]
    target = images[img_i]
    target = torch.Tensor(target).to(device)

    # 이미지로부터 ray 받아오기
    rays_o, rays_d = hp.get_rays(H, W, K, torch.Tensor(pose))
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
    hp.render(H, W, K, rays=batch_rays, near=near, far=far, use_viewdirs=True)