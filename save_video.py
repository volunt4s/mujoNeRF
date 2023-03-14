import os
import torch
import imageio
import warnings
import numpy as np

import src.nerf.render_helper as hp
from src.nerf.load_nerf_data import load_mujoco_data
from src.nerf.config_parse import config_parser

# To avoid pytorch meshgrid warning
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_video():
    parser = config_parser()
    args = parser.parse_args()

    images, poses, render_poses, hwf, i_split = load_mujoco_data(video_ref_id="20")
    H, W, focal = hwf
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
        ])

    _, render_kwargs_test, _, _, _ = hp.create_nerf(args)
    bds_dict = {
        'near' : args.near,
        'far' : args.far,
    }
    render_kwargs_test.update(bds_dict)

    # Create rendered image save directory
    basedir = args.basedir
    expname = args.expname
    save_dir = os.path.join(basedir, expname, "rendered")
    os.makedirs(save_dir, exist_ok=True)

    print("Load trained state dict")
    ckpt = torch.load(os.path.join(basedir, expname, "trained_model.tar"))
    render_kwargs_test['network_fn'].load_state_dict(ckpt['network_fn_state_dict'])
    if render_kwargs_test['network_fine'] is not None:
        render_kwargs_test['network_fine'].load_state_dict(ckpt['network_fine_state_dict'])

    print("Start")
    with torch.no_grad():
        rgbs, disps = hp.render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, savedir=save_dir)
        imageio.mimwrite(os.path.join(save_dir, 'rgb_video.mp4'), hp.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(save_dir, 'depth_video.mp4'), hp.to8b(disps / np.max(disps)), fps=30, quality=8)

    print('Video and rendered image saved at : ', save_dir)


if __name__ == "__main__":
    # Set all tensor to cuda if using GPU
    if device != "cpu":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    save_video()