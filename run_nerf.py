import os
import numpy as np
import time
import warnings
import torch
from tqdm import tqdm, trange

import src.nerf.render_helper as hp
from src.nerf.config_parse import config_parser
from src.nerf.load_nerf_data import load_mujoco_data

# Avoid pytorch meshgrid warning
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

# TODO: logs 수정, parser argument 수정, 막혀진벽, stl 파일, 리드미


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    images, poses, render_poses, hwf, i_split = load_mujoco_data(video_ref_id=args.video_ref_id)
    print('Loaded mujoco custom ', images.shape, render_poses.shape, hwf)
    i_train, i_test, i_val = i_split

    # Ray bound
    near = 1.0
    far = 6.0

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    print("Args, config saved at : ", os.path.join(basedir, expname))
    
    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = hp.create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : args.near,
        'far' : args.far,
    }

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    N_iters = 10000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    start = start + 1

    for i in trange(start, N_iters):
        time0 = time.time()

        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3,:4]

        if N_rand is not None:
            pose = torch.Tensor(pose)
            rays_o, rays_d = hp.get_rays(H, W, K, pose)  # (H, W, 3), (H, W, 3)

            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = hp.render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = hp.img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = hp.mse2psnr(img_loss)

        # Compute Fine model loss
        if 'rgb0' in extras:
            img_loss0 = hp.img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = hp.mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # Update learning rate
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1

    print("Save trained model")
    trained_path = os.path.join(os.getcwd(), "trained")
    os.makedirs(trained_path, exist_ok=True)

    torch.save({
        'global_step' : global_step,
        'network_fn_state_dict' : render_kwargs_train['network_fn'].state_dict(),
        'network_fine_state_dict' : render_kwargs_train['network_fine'].state_dict(),
        'optimizer_state_dict' : optimizer.state_dict()
    }, os.path.join(trained_path, "model.tar"))

    print("Done")

if __name__=='__main__':
    # Set all tensor to cuda
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
