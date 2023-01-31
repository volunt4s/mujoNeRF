from mujoco import viewer
from src.models.world import MujocoWorldBase
from src.models.arenas import TableArena
from src.models.objects import BottleObject
from src.models.objects import MilkObject
from src.models.objects import CanObject


def main():
    world = MujocoWorldBase()
    table_arena = TableArena()
    # bottle_object = BottleObject(pos=[0, 0, 0.9])
    milk_object = MilkObject(pos=[0, 0, 0.9])
    can_object = CanObject(pos=[0, 0, 0.9])


    world.merge(table_arena)
    world.merge(can_object, object_type="object")

    model = world.get_model()
    viewer.launch(model)

if __name__ == "__main__":
    main()
