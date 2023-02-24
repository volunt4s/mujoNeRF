from src.generate.load_nerf_data import load_custom_data
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    imgs, poses, [H, W, focal], i_split = load_custom_data()
    
    # TODO : 기존과 같은 효과 보려면 transpose후 곱해야한다. 이유 살펴보기
    rot_vec = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    translation = poses[:, :3, -1]

    # 00 : <camera pos="1.500 0.000 1.500" xyaxes="-0.000 1.000 0.000 -0.591 -0.000 0.806"/>
    # 01 : <camera pos="1.444 0.405 1.500" xyaxes="-0.270 0.963 0.000 -0.569 -0.160 0.806"/>
    # 02 : <camera pos="1.282 0.779 1.500" xyaxes="-0.520 0.854 -0.000 -0.505 -0.307 0.806"/>

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