from src.models.world import MujocoWorldBase
from src.models.arenas import TableArena
from src.models.objects import CanObject
from src.models.camera import CameraSet

from dm_control import mujoco
from mujoco import viewer

import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    world = MujocoWorldBase()
    table_arena = TableArena()
    can_object = CanObject(pos=[0, 0, 0.9])
    
    camera_set = CameraSet(
        ref_pos=np.array([1.500, 0.000, 1.500]),
        ref_xyaxes=np.array([0.000, 1.000, 0.000, -0.591, 0.005, 0.806]),
        y_target_angle=[0, -np.pi/7],
        y_times=6,
        z_target_angle=[0, 2*np.pi],
        z_times=24)

    camera_set_xml = camera_set.get_camera_xml_lst
    
    world.merge(table_arena)
    world.merge(can_object)
    world.merge_camera_set(camera_set_xml)
    
    model = world.get_model(mode="mujoco")
    physics = world.get_model(mode="dm_control")
    
    base_dir = os.getcwd() + "/nerf_data"
    camera_set.generate_nerf_data(physics, base_dir)
    # viewer.launch(model)

if __name__ == "__main__":
    main()