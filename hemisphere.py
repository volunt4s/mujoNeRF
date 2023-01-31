import numpy as np
import xml.etree.ElementTree as ET

def get_rotated_camera_about_zaxis(thetas, pos, xyaxes):
    if not isinstance(thetas, np.ndarray):
        raise ValueError("Rotation angles are must defined by ndarray")

    x_axis = xyaxes[:3]
    y_axis = xyaxes[3:]
    xml_elem_lst = []

    for i, theta in enumerate(thetas):
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        roted_pos = np.matmul(rot, pos)
        roted_x_axis = np.matmul(rot, x_axis)
        roted_y_axis = np.matmul(rot, y_axis)
        roted_xyaxes = np.concatenate([roted_x_axis, roted_y_axis])
        roted_xml_elem = get_camera_xml_element(mode="fixed", name=str(i), pos=roted_pos, xyaxes=roted_xyaxes)
        xml_elem_lst.append(roted_xml_elem)

    return xml_elem_lst

def get_camera_xml_element(mode, name, pos, xyaxes):
    camera = ET.Element("camera")
    camera.attrib["mode"] = mode
    camera.attrib["name"] = name
    camera.attrib["pos"] = " ".join(map(str, pos))
    camera.attrib["xyaxes"] = " ".join(map(str, xyaxes))
    return camera
    
if __name__ == "__main__":
    pass
