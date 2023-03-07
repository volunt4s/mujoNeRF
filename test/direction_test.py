from src.nerf.load_nerf_data import load_custom_data
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    imgs, poses, [H, W, focal], i_split = load_custom_data()
    rot_vec = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    translation = poses[:, :3, -1]
    print(poses[0])

    ax = plt.figure().add_subplot(projection="3d")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    ax.quiver(
        translation[1:, 0].flatten(),
        translation[1:, 1].flatten(),
        translation[1:, 2].flatten(),
        rot_vec[1:, 0].flatten(),
        rot_vec[1:, 1].flatten(),
        rot_vec[1:, 2].flatten(), length=0.1, normalize=True)

    print(translation[0])
    print(rot_vec[0])

    ax.quiver(
        translation[0][0].flatten(),
        translation[0][1].flatten(),
        translation[0][2].flatten(),
        rot_vec[0][0].flatten(),
        rot_vec[0][1].flatten(),
        rot_vec[0][2].flatten(), length=0.2, normalize=True, color='r')

    plt.show()