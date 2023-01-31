from src.models.world import MujocoWorldBase
from src.models.arenas import TableArena
from src.models.objects import BottleObject
from src.models.objects import MilkObject
from src.models.objects import CanObject

from dm_control import mujoco

import matplotlib.pyplot as plt
import numpy as np


def main():
    world = MujocoWorldBase()
    table_arena = TableArena()
    milk_object = MilkObject(pos=[0, 0, 0.9])
    can_object = CanObject(pos=[0, 0, 0.9])

    world.merge(table_arena)
    world.merge(can_object, object_type="object")

    physics = world.get_model()    

    pixels = physics.render(camera_id="frontview")
    plt.imshow(pixels)
    plt.show()
    camera = mujoco.Camera(physics, camera_id="birdview")
    camera_matrces = camera.matrices()
    print(camera_matrces.translation)
    print(camera_matrces.rotation)
    print(camera_matrces.focal)

    
if __name__ == "__main__":
    main()