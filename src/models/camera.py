import numpy as np
import xml.etree.ElementTree as ET

class CameraSet:
    def __init__(self, ref_pos, ref_xyaxes):
        self.ref_pos = ref_pos
        self.ref_xyaxes = ref_xyaxes

    def get_hemisphere_camera_samples(self, y_target_angle, y_times, z_target_angle, z_times):
        xml_elem_lst = []
        y_roted_lst = self.get_rotated_camera_about_zyaxis(y_target_angle, y_times, self.ref_pos, self.ref_xyaxes, axis="y")
        
        for i, [y_roted_pos, y_roted_xyaxes] in enumerate(y_roted_lst):
            z_roted_lst = self.get_rotated_camera_about_zyaxis(z_target_angle, z_times, y_roted_pos, y_roted_xyaxes, axis="z")
            for j, [z_roted_pos, z_roted_xyaxes] in enumerate(z_roted_lst):
                rot_name = str(i) + str(j)
                roted_xml_elem = self.get_xml_element(mode="fixed", name=rot_name, pos=z_roted_pos, xyaxes=z_roted_xyaxes)
                xml_elem_lst.append(roted_xml_elem)
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