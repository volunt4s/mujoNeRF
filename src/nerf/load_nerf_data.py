import os
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def pose_zrot(angle, ref):
    rot = lambda theta: np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    c2w = torch.Tensor(rot(angle) @ ref)
    return c2w


def load_mujoco_data(video_ref_id):
    base_dir = os.path.join(os.getcwd(), "nerf_data")
    json_dir = os.path.join(base_dir, "data.json")

    cam_ids = []
    imgs = []
    poses = []

    with open(json_dir, 'r') as f:
        meta = json.load(f)
    
    focal = meta["focal_length"]

    for frame in meta["frames"]:
        fname = frame["file_dir"]
        try:
            imgs.append(imageio.v3.imread(fname))
        except FileNotFoundError:
            print("FileNotFoundError: Please generate NeRF data first.")
            exit(0)
        cam_ids.append(frame["camera_id"])
        poses.append(frame["transform_matrix"])

    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    
    # Get reference render pose
    video_ref_idx = cam_ids.index(video_ref_id)    
    video_ref_pose = poses[video_ref_idx]

    H, W = imgs[0].shape[:2]

    # Shuffle dataset index
    img_cnt = imgs.shape[0]
    idx_lst = np.arange(img_cnt)
    np.random.shuffle(idx_lst)
    train_idx = idx_lst[0:img_cnt-30]
    test_idx = idx_lst[img_cnt-30:img_cnt-10]
    val_idx = idx_lst[img_cnt-10:]

    render_poses = torch.stack([pose_zrot(angle, video_ref_pose) for angle in np.linspace(0.0, 2*np.pi, 40)], 0)
    i_split = [train_idx, test_idx, val_idx]

    return imgs, poses, render_poses, [H, W, focal], i_split