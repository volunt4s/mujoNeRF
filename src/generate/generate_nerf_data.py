from src.models.world import MujocoWorldBase
from src.models.arenas import TableArena
from src.models.objects import CanObject
from src.models.objects import CatObject

from src.models.camera import CameraSet
from dm_control import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import numpy as np
import os

# <camera pos="1.825 0.028 1.551" xyaxes="0.024 1.000 -0.000 -0.468 0.011 0.884"/>

def main():
    world = MujocoWorldBase()
    table_arena = TableArena()

    cat_object = CatObject()
    camera_set = CameraSet(
        ref_pos=np.array([0.759, 0.018, 1.309]),
        ref_xyaxes=np.array([0.009, 1.000, 0.000, -0.510, 0.005, 0.860]),
        y_target_angle=[0, -np.pi/5],
        y_times=4,
        z_target_angle=[0, 2*np.pi],
        z_times=24)

    camera_set_xml = camera_set.get_camera_xml_lst
    
    world.merge(table_arena)
    world.merge(cat_object)
    world.merge_camera_set(camera_set_xml)
    
    model = world.get_model(mode="mujoco")
    physics = world.get_model(mode="dm_control")
    
    camera_set.generate_nerf_data(physics)


if __name__ == "__main__":
    main()