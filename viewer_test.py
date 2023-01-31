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
    camera_set_xml = camera_set.get_rotated_camera_about_zaxis(2 * np.pi, 30)

    world.merge(table_arena)
    world.merge(can_object)
    world.merge_camera_set(camera_set_xml)

    print(world.get_xml())

    model = world.get_model(mode="mujoco")
    physics = world.get_model(mode="dm_control")
    

    viewer.launch(model)

if __name__ == "__main__":
    main()

    # front
    # <camera pos="1.500 0.000 1.500" xyaxes="0.000 1.000 0.000 -0.591 0.005 0.806"/>

    # left side
    # <camera pos="0.000 1.500 1.500" xyaxes="1.000 0.000 0.000 0.010 0.557 0.830"/>

    # back
    # <camera pos="-1.500 0.000 1.500" xyaxes="-0.020 -1.000 -0.000 0.548 -0.011 0.836"/>

    # right side
    # <camera pos="0.000 1.500 1.500" xyaxes="-1.000 0.011 -0.000 -0.006 -0.559 0.829"/>
