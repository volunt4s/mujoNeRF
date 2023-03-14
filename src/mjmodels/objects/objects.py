import copy
import xml.etree.ElementTree as ET
import numpy as np
from src.mjmodels.base import MujocoXML
from src.utils.mjcf_utils import string_to_array, array_to_string, new_joint, new_geom
from src.utils.transform_utils import euler2mat, mat2quat

class MujocoObject:

    def __init__(self):
        pass

    @property
    def body_xpos(self):
        raise NotImplementedError

    @property
    def body_ori(self):
        raise NotImplementedError

    def get_collision(self):
        raise NotImplementedError

    def set_color(self, rgba):
        raise NotImplementedError


class MujocoXMLObject(MujocoXML, MujocoObject):
    """
    MujocoObjects that are loaded from xml files
    """

    def __init__(
        self, 
        fname, 
        name, 
        pos, 
        rot, 
        joints
    ):

        MujocoXML.__init__(self, fname)

        self.name = name
        self._joints = joints

        if np.array(rot).shape == (3,):
            rot = mat2quat(euler2mat(rot))

        self._body_object = self.worldbody.find("./body")
        self._bottom_site = self.worldbody.find("./body/site[@name='bottom_site']")
        self._top_site = self.worldbody.find("./body/site[@name='top_site']")
        self._horizontal_radius_site = self.worldbody.find("./body/site[@name='horizontal_radius_site']")
        self._collision = self.worldbody.find("./body/body[@name='collision']")
        self._visual = self.worldbody.find("./body/body[@name='visual']")

        self._body_object.set("pos", array_to_string(pos))
        self._body_object.set("quat", array_to_string(rot))

        self._body_size = None

    @property
    def bottom_offset(self):
        return string_to_array(self._bottom_site.get("pos"))

    @property
    def top_offset(self):
        return string_to_array(self._top_site.get("pos"))

    @property
    def horizontal_radius(self):
        return string_to_array(self._horizontal_radius_site.get("pos"))[0]

    @property
    def body_object(self):
        return self._body_object

    @body_object.setter
    def body_object(self, body_name):
        assert type(body_name) == str
        self._body_object = self.worldbody.find("{}".format(body_name))

    @property
    def body_xpos(self):
        return string_to_array(self._body_object.get("pos"))
        
    @body_xpos.setter
    def body_xpos(self, pos):
        self._body_object.set("pos", array_to_string(pos))

    @property
    def body_ori(self):
        return string_to_array(self._body_object.get("quat"))
        
    @body_ori.setter
    def body_ori(self, rot):
        assert np.array(rot) == (3,), "Orientation type is Euler!!"
        rot = mat2quat(euler2mat(rot))
        self._body_object.set("quat", array_to_string(rot))

    @property
    def body_size(self):
        return string_to_array(self._body_object.get("size"))

    @body_size.setter
    def body_size(self, size):
        self._body_size = self._body_object.set("size", array_to_string(size))

    def get_collision(self):
        collision = copy.deepcopy(self.worldbody.find("./body/body[@name='collision']"))
        collision.attrib.pop("name")
        col_name = self.name+"_col"

        geoms = collision.findall("geom")
        duplicate_geoms = copy.deepcopy(geoms)
        if self.name is not None:
            collision.attrib["name"] = col_name
            if len(geoms) == 1:
                geoms[0].set("name", col_name+"-0")
            else:
                for i in range(len(geoms)):
                    geoms[i].set("name", "{}-{}".format(col_name, i))
        
        geom_group = duplicate_geoms[0].get("group")
        duplicate_geoms[0].set("group", "1")
        
        if int(geom_group) == 1:
            duplicate_geoms[0].set("group", "0")
        
        collision.append(ET.Element("geom", attrib=duplicate_geoms[0].attrib))
        
        collision.set("pos", array_to_string(self.body_xpos))
        collision.set("quat", array_to_string(self.body_ori))

        if self._joints is not None:
            collision.append(new_joint(name=col_name+"_joint", **self._joints[0]))
        return collision

    def set_color(self, rgba:np.ndarray, geom_type="collision"):
        if geom_type == "collision":
            geom = self._collision.find("geom")
        
        if geom_type == "visual":
            geom = self._visual.find("geom")

        geom.set("rgba", array_to_string(rgba))