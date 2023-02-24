from src.models.world import MujocoWorldBase
from src.models.arenas import TableArena
from src.models.arenas import EmptyArena
from src.models.objects import CanObject
from src.models.objects import CatObject
from src.models.camera import CameraSet
from dm_control import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    world = MujocoWorldBase()
    table_arena = TableArena()
    empty_arna = EmptyArena()

    cat_object = CatObject()
    
    camera_set = CameraSet(
        ref_pos=np.array([2.070, 0.018, 0.282]),
        ref_xyaxes=np.array([-0.015, 1.000, 0.000, -0.171, -0.002, 0.985]),
        y_target_angle=[0, -np.pi/4],
        y_times=5,
        z_target_angle=[0, 2*np.pi],
        z_times=24)
    
    camera_set_xml = camera_set.get_camera_xml_lst
    
    world.merge(empty_arna)
    world.merge(cat_object)
    world.merge_camera_set(camera_set_xml)
    
    model = world.get_model(mode="mujoco")
    physics = world.get_model(mode="dm_control")
    
    # viewer.launch(model)
    camera_set.generate_nerf_data(physics)

if __name__ == "__main__":
    main()
