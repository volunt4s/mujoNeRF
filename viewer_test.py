from src.models.world import MujocoWorldBase
from src.models.arenas import TableArena
from src.models.objects import BottleObject
from src.models.objects import MilkObject
from src.models.objects import CanObject
from src.models.camera import CameraSet

from dm_control import mujoco
from mujoco import viewer

import matplotlib.pyplot as plt
import numpy as np


def main():
    world = MujocoWorldBase()
    table_arena = TableArena()
    milk_object = MilkObject(pos=[0, 0, 0.9])
    can_object = CanObject(pos=[0, 0, 0.9])
    
    camera_set = CameraSet(ref_pos=np.array([1.500, 0.000, 1.500]), ref_xyaxes=np.array([0.000, 1.000, 0.000, -0.591, 0.005, 0.806]))
    camera_set_xml = camera_set.get_hemisphere_camera_samples([0, -np.pi/7], 6, [0, 2*np.pi], 24)

    world.merge(table_arena)
    world.merge(can_object)
    world.merge_camera_set(camera_set_xml)
    
    model = world.get_model(mode="mujoco")
    physics = world.get_model(mode="dm_control")
    
    camera_id_lst = []
    for i in range(6):
        for j in range(24):
            camera_id_lst.append(str(i)+str(j))
    
    mj_camera_lst = []
    for camera_id in camera_id_lst:
        mj_camera = mujoco.Camera(physics, camera_id=camera_id)
        mj_camera_lst.append(mj_camera)

    cam1 = mj_camera_lst[94]

    plt.imshow(cam1.render())
    plt.show()

if __name__ == "__main__":
    main()