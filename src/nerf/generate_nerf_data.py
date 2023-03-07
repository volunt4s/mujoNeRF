import numpy as np
from mujoco import viewer

from src.models.world import MujocoWorldBase
from src.models.arenas import EmptyArena
from src.models.objects import CatObject
from src.models.camera import CameraSet


def main():
    world = MujocoWorldBase()
    empty_arena = EmptyArena()
    cat_object = CatObject()

    camera_set = CameraSet(
        ref_pos=np.array([2.704, -0.032, 0.444]),
        ref_xyaxes=np.array([0.012, 1.000, 0.000, -0.162, 0.002, 0.987]),
        y_target_angle=[0, -np.pi/3],
        y_times=7,
        z_target_angle=[0, 2*np.pi],
        z_times=24)

    camera_set_xml = camera_set.get_camera_xml_lst
    
    world.merge(empty_arena)
    world.merge(cat_object)
    world.merge_camera_set(camera_set_xml)
    
    model = world.get_model(mode="mujoco")
    physics = world.get_model(mode="dm_control")
    
    # First, get proper reference camera pose in Mujoco gui viewer 
    # viewer.launch(model)
    
    # Then, generate image data
    camera_set.generate_nerf_data(physics)


if __name__ == "__main__":
    main()