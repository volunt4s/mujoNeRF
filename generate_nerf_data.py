import numpy as np
from mujoco import viewer

from src.mjmodels.world import MujocoWorldBase
from src.mjmodels.arenas import EmptyArena
from src.mjmodels.objects import DragonObject
from src.mjmodels.camera import CameraSet


def main():
    world = MujocoWorldBase()
    empty_arena = EmptyArena()
    dragon_object = DragonObject()

    camera_set = CameraSet(
        xml_string='<camera pos="2.947 -0.016 0.318" xyaxes="0.006 1.000 -0.000 -0.107 0.001 0.994"/>',
        y_target_angle=[0, -np.pi/3],
        y_times=7,
        z_target_angle=[0, 2*np.pi],
        z_times=16)

    camera_set_xml = camera_set.get_camera_xml_lst
    
    world.merge(empty_arena)
    world.merge(dragon_object)
    world.merge_camera_set(camera_set_xml)
    
    model = world.get_model(mode="mujoco")
    physics = world.get_model(mode="dm_control")
    
    # First, get proper reference camera pose in Mujoco gui viewer 
    viewer.launch(model)
    
    # Then, generate image data
    # camera_set.generate_nerf_data(physics)


if __name__ == "__main__":
    main()