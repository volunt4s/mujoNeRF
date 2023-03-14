import configargparse
import numpy as np
from mujoco import viewer

from src.mjmodels.world import MujocoWorldBase
from src.mjmodels.arenas import EmptyArena
from src.mjmodels.objects import CatObject
from src.mjmodels.camera import CameraSet


def config_parser():
    parser = configargparse.ArgumentParser()
    true_false_list = ['true', 'yes', "1", 't','y']
    parser.add_argument('--run_gui', type=lambda s: s.lower() in true_false_list, required=False, default=False,
                        help='run mujoco gui')
    parser.add_argument("--cam_xml", type=str, default=None,
                        help='reference camera xml string')
    parser.add_argument("--generate", type=lambda s: s.lower() in true_false_list, required=False, default=False,
                        help='generate nerf dataset')
    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()

    # Initalize mujoco arena and object
    world = MujocoWorldBase()
    empty_arena = EmptyArena()
    cat_object = CatObject()
    
    # Merge arena, object, camera to base world
    world.merge(empty_arena)
    world.merge(cat_object)

    if args.cam_xml is not None:
        # Initialize camera sets
        camera_set = CameraSet(
            xml_string=args.cam_xml,
            y_target_angle=[0, -np.pi/3],
            y_times=7,
            z_target_angle=[0, 2*np.pi],
            z_times=16)
        camera_set_xml = camera_set.get_camera_xml_lst
        world.merge_camera_set(camera_set_xml)
    
    model = world.get_model(mode="mujoco")
    physics = world.get_model(mode="dm_control")
    
    # First, get proper reference camera pose in Mujoco gui viewer 
    if args.run_gui:
        viewer.launch(model)
    
    # Then, generate image data
    if args.generate:
        camera_set.generate_nerf_data(physics)


if __name__ == "__main__":
    main()