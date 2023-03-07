import numpy as np
from mujoco import viewer

from src.models.world import MujocoWorldBase
from src.models.arenas import EmptyArena
from src.models.objects import CatObject
from src.models.camera import CameraSet

from src.models.objects import LegoObject
# <camera pos="1.913 -0.013 0.531" xyaxes="0.007 1.000 0.000 -0.267 0.002 0.964"/>
# <camera pos="-0.219 -2.234 0.674" xyaxes="1.000 0.003 -0.000 -0.001 0.280 0.960"/>
# <camera pos="2.202 0.017 0.475" xyaxes="-0.008 1.000 0.000 -0.211 -0.002 0.977"/>

def main():
    world = MujocoWorldBase()
    empty_arena = EmptyArena()
    # cat_object = CatObject()
    lego_object = LegoObject()

    # TODO : data auto parsing
    camera_set = CameraSet(
        ref_pos=np.array([2.202, 0.017, 0.475]),
        ref_xyaxes=np.array([-0.008, 1.000, 0.000, -0.211, -0.002, 0.977]),
        y_target_angle=[0, -np.pi/3],
        y_times=7,
        z_target_angle=[0, 2*np.pi],
        z_times=16)

    camera_set_xml = camera_set.get_camera_xml_lst
    
    world.merge(empty_arena)
    world.merge(lego_object)
    # world.merge(cat_object)
    world.merge_camera_set(camera_set_xml)
    
    model = world.get_model(mode="mujoco")
    physics = world.get_model(mode="dm_control")
    
    # First, get proper reference camera pose in Mujoco gui viewer 
    # viewer.launch(model)
    
    # Then, generate image data
    camera_set.generate_nerf_data(physics)


if __name__ == "__main__":
    main()