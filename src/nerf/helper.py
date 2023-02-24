import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.nerf.model import Embedder
from src.nerf.model import NeRF

device = "cpu"

def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], dim=-1)
    
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims'    : 3,
                'max_freq_log2' : multires-1,
                'num_freqs'     : multires,
                'log_sampling'  : True,
                'periodic_fns'  : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


def create_nerf(args):
    #TODO : args 텍스트 형식으로 받아오기


    # 임베더 받아오기 -> 포인트, 방향 둘 다
    embed_fn, input_ch = get_embedder(args['multires'], args['i_embed'])
    embeddirs_fn = None
    input_ch_views = 0
    if args['use_viewdirs']:
        embeddirs_fn, input_ch_views = get_embedder(args['multires_views'], args['i_embed'])
    output_ch = 5 if args['N_importance'] > 0 else 4
    
    # 모델 정의
    model = NeRF(
        D=args['netdepth'], W=args['netwidth'],
        input_ch=input_ch, output_ch=output_ch,
        input_ch_views=input_ch_views, use_viewdirs=args['use_viewdirs']
    ).to(device)
    grad_vars = list(model.parameters())
    
    # Fine 모델 정의
    model_fine = None
    if args['N_importance'] > 0:
        model_fine = NeRF(
            D=args["netdepth_fine"], W=args["netwidth_fine"],
            input_ch=input_ch, output_ch=output_ch,
            input_ch_views=input_ch_views, use_viewdirs=args["use_viewdirs"]
        ).to(device)
        grad_vars += list(model_fine.parameters())
    
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args['netchunk']
    )
    
    # 옵티마이저 정의
    optimizer = torch.optim.Adam(params=grad_vars, lr=args["lrate"], betas=(0.9, 0.999))

    # 스타트 인덱스
    start = 0

    # nerf 초기화 dict
    render_kwargs_train = {
        'network_query_fn'  : network_query_fn,
        'perturb'           : args["perturb"],
        'N_importance'      : args["N_importance"],
        'network_fine'      : model_fine,
        'N_samples'         : args["N_samples"],
        'network_fn'        : model,
        'use_viewdirs'      : args["use_viewdirs"],
        'white_bkgd'        : args["white_bkgd"],
        'raw_noise_std'     : args["raw_noise_std"]
    }

    return render_kwargs_train, start, grad_vars, optimizer

###########################################################################################
#### volume rendering helper function
###########################################################################################

def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def render(H, W, K, rays=None, use_viewdirs=True, near=0., far=1., chunk=1024*32, retraw=True, **kwargs):
    rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    
    batchify_rays(rays, chunk, **kwargs)


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        break


def render_rays(
        ray_batch,
        network_fn,
        network_query_fn,
        N_samples,
        retraw=False,
        lindisp=False,
        perturb=0.0,
        N_importance=0,
        network_fine=None,
        white_bkgd=False,
        raw_noise_std=0.0):
    
    N_rays = ray_batch.shape[0] # 1024개
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    # lindisp is always False
    z_vals = near * (1.0-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])
    # perturb is always 1.0
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)

    t_rand = torch.rand(z_vals.shape)

    z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)

    # Hierarchicar sampling
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        #TODO :


def raw2outputs(raw, z_vals, rays_d):
    raw2alpha = lambda raw, dists, act_fn=F.relu : 1.0 - torch.exp(-act_fn(raw)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], dim=-1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.0

    alpha = raw2alpha(raw[..., 3] + noise, dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.0-alpha+1e-10], dim=-1), dim=-1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

    depth_map = torch.sum(weights * z_vals, dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, dim=-1))
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, disp_map, acc_map, weights, depth_map

