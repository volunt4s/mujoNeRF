import numpy as np
import xml.etree.ElementTree as ET
import os
import json
import imageio
from tqdm import tqdm
from dm_control import mujoco
from scipy.spatial.transform import Rotation as R

class CameraSet:
    def __init__(self, ref_pos, ref_xyaxes, y_target_angle, y_times, z_target_angle, z_times):
        self.ref_pos = ref_pos
        self.ref_xyaxes = ref_xyaxes
        self.camera_id_lst = []
        self.camera_xml_lst = self.get_hemisphere_camera_xml(y_target_angle, y_times, z_target_angle, z_times)

    def generate_nerf_data(self, physics):
        print("Generating image files and JSON data...")
        base_dir = os.path.join(os.getcwd(),  "nerf_data")
        json_frames = []
        json_dir = os.path.join(base_dir, "data.json")
        img_dir = os.path.join(base_dir, "img/")
        # Generate proper directory
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        for camera_id in tqdm(self.camera_id_lst):
            mj_camera = mujoco.Camera(physics, camera_id=camera_id)
            # JSON format processing
            rotation_matrix = mj_camera.matrices().rotation.transpose()
            ################# check rot mat transpose training
            
            translation = mj_camera.matrices().translation
            focal_length = mj_camera.matrices().focal
            transform_mat, focal = self.preprocess_camera_params(rotation_matrix, translation, focal_length)
            json_frames.append(self.get_frame_json_element(camera_id, transform_mat, base_dir))
            # Image save
            img = mj_camera.render()
            imageio.imwrite(img_dir + str(camera_id) + ".png", img)
        
        # JSON save
        json_object = {
            "focal_length" : focal,
            "frames" : json_frames
        }
        with open(json_dir, "w") as f:
            json.dump(json_object, f, indent=4)
        
        print("Data generated at " + base_dir)

    def get_frame_json_element(self, camera_id, transform_mat, base_dir):
        img_dir = base_dir + "/img/" + str(camera_id) + ".png"
        frame_json_elem = {
            "camera_id" : camera_id,
            "file_dir" : img_dir,
            "transform_matrix" : transform_mat
        }
        return frame_json_elem

    def preprocess_camera_params(self, rotation_matrix, translation, focal_length):
        # Transformation matrix processing
        transform_mat = np.zeros_like(rotation_matrix)
        transform_mat[:3, :3] = rotation_matrix[:3, :3]
        transform_mat[:3, -1] = translation[:3, -1] * -1
        transform_mat[3, 3] = 1.0
        transform_mat = transform_mat.tolist()
        # Focal length processing
        if abs(focal_length[0, 0]) == abs(focal_length[1, 1]):
            focal = abs(focal_length[0, 0])
        return transform_mat, focal

    def get_hemisphere_camera_xml(self, y_target_angle, y_times, z_target_angle, z_times):
        xml_elem_lst = []
        y_roted_lst = self.get_rotated_camera_about_zyaxis(y_target_angle, y_times, self.ref_pos, self.ref_xyaxes, axis="y")
        
        for i, [y_roted_pos, y_roted_xyaxes] in enumerate(y_roted_lst):
            z_roted_lst = self.get_rotated_camera_about_zyaxis(z_target_angle, z_times, y_roted_pos, y_roted_xyaxes, axis="z")
            for j, [z_roted_pos, z_roted_xyaxes] in enumerate(z_roted_lst):
                rot_name = str(i) + str(j)
                roted_xml_elem = self.get_xml_element(mode="fixed", name=rot_name, pos=z_roted_pos, xyaxes=z_roted_xyaxes)
                xml_elem_lst.append(roted_xml_elem)
                self.camera_id_lst.append(rot_name)
        return xml_elem_lst

    def get_rotated_camera_about_zyaxis(self, target_angle, times, pos, xyaxes, axis):
        thetas = np.linspace(target_angle[0], target_angle[1], times)
        x_axis = xyaxes[:3]
        y_axis = xyaxes[3:]
        roted_lst = []

        if axis == "z":
            rot = lambda theta: np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
        elif axis == "y":
            rot = lambda theta: np.array([
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]
                ])
        else:
            raise ValueError("Input proper rotation axis (possible : z, y)")

        for theta in thetas:
            roted_pos = np.matmul(rot(theta), pos)
            roted_x_axis = np.matmul(rot(theta), x_axis)
            roted_y_axis = np.matmul(rot(theta), y_axis)
            roted_xyaxes = np.concatenate([roted_x_axis, roted_y_axis])
            roted_lst.append([roted_pos, roted_xyaxes])
        return roted_lst

    def get_xml_element(self, mode, name, pos, xyaxes):
        camera = ET.Element("camera")
        camera.attrib["mode"] = mode
        camera.attrib["name"] = name
        camera.attrib["pos"] = " ".join(map(str, pos))
        camera.attrib["xyaxes"] = " ".join(map(str, xyaxes))
        return camera

    @property
    def get_camera_id_lst(self):
        return self.camera_id_lst
    
    @property
    def get_camera_xml_lst(self):
        return self.camera_xml_lst