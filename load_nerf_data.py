import os
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt

# img : (400, 400, 400, 4)
# poses : (400, 4, 4)
# [H, W, focal] : [400, 400, 555.5555155968841]

def load_data():
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
        imgs.append(imageio.v3.imread(fname))
        cam_ids.append(frame["camera_id"])
        poses.append(frame["transform_matrix"])
        
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    H, W = imgs[0].shape[:2]

    #TODO : half resolution processing

    return imgs, poses, [H, W, focal]