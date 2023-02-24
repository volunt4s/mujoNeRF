from load_nerf_data import load_custom_data
import numpy as np

if __name__ == "__main__":
    imgs, poses, [H, W, focal], i_split = load_custom_data()
    idx_lst = np.arange(imgs.shape[0])
    np.random.shuffle(idx_lst)
    train_idx = idx_lst[0:110]
    test_idx = idx_lst[110:130]
    val_idx = idx_lst[130:]
