# utility functions for manipulating MJCF XML models

import xml.etree.ElementTree as ET
import os
sim_path = os.path.abspath(os.path.dirname(__file__)+"/../" )
import numpy as np

import src.models

RED = [1, 0, 0, 1]
GREEN = [0, 1, 0, 1]
BLUE = [0, 0, 1, 1]
GRIPPER_COLLISION_COLOR = [0, 0, 0.5, 1]
OBJECT_COLLISION_COLOR = [0.5, 0, 0, 1]

def xml_path_completion(xml_path):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package
    """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        full_path = os.path.join(src.models.assets_root, xml_path)
    return full_path


def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"
    """
    return " ".join(["{}".format(x) for x in array])


def string_to_array(string):
    """
    Converts a array string in mujoco xml to np.array.

    Examples:
        "0 1 2" => [0, 1, 2]
    """
    return np.array([float(x) for x in string.split(" ")])


def set_alpha(node, alpha=0.1):
    """
    Sets all a(lpha) field of the rgba attribute to be @alpha
    for @node and all subnodes
    used for managing display
    """
    for child_node in node.findall(".//*[@rgba]"):
        rgba_orig = string_to_array(child_node.get("rgba"))
        child_node.set("rgba", array_to_string(list(rgba_orig[0:3]) + [alpha]))


def new_joint(**kwargs):
    """
    Creates a joint tag with attributes specified by @**kwargs.
    """

    element = ET.Element("joint", attrib=kwargs)
    return element


def new_actuator(joint, act_type="actuator", **kwargs):
    """
    Creates an actuator tag with attributes specified by @**kwargs.

    Args:
        joint: type of actuator transmission.
            see all types here: http://mujoco.org/book/modeling.html#actuator
        act_type (str): actuator type. Defaults to "actuator"

    """
    element = ET.Element(act_type, attrib=kwargs)
    element.set("joint", joint)
    return element


def new_site(name, rgba=RED, pos=(0, 0, 0), size=(0.005,), **kwargs):
    """
    Creates a site element with attributes specified by @**kwargs.

    Args:
        name (str): site name.
        rgba: color and transparency. Defaults to solid red.
        pos: 3d position of the site.
        size ([float]): site size (sites are spherical by default).
    """
    kwargs["rgba"] = array_to_string(rgba)
    kwargs["pos"] = array_to_string(pos)
    kwargs["size"] = array_to_string(size)
    kwargs["name"] = name
    element = ET.Element("site", attrib=kwargs)
    return element


def new_geom(geom_type, size, pos=(0, 0, 0), rgba=RED, group=0, **kwargs):
    """
    Creates a geom element with attributes specified by @**kwargs.

    Args:
        geom_type (str): type of the geom.
            see all types here: http://mujoco.org/book/modeling.html#geom
        size: geom size parameters.
        pos: 3d position of the geom frame.
        rgba: color and transparency. Defaults to solid red.
        group: the integrer group that the geom belongs to. useful for
            separating visual and physical elements.
    """
    kwargs["type"] = str(geom_type)
    kwargs["size"] = array_to_string(size)
    kwargs["rgba"] = array_to_string(rgba)
    kwargs["group"] = str(group)
    kwargs["pos"] = array_to_string(pos)
    element = ET.Element("geom", attrib=kwargs)
    return element


def new_body(name=None, pos=None, **kwargs):
    """
    Creates a body element with attributes specified by @**kwargs.

    Args:
        name (str): body name.
        pos: 3d position of the body frame.
    """
    if name is not None:
        kwargs["name"] = name
    if pos is not None:
        kwargs["pos"] = array_to_string(pos)
    element = ET.Element("body", attrib=kwargs)
    return element


def new_inertial(name=None, pos=(0, 0, 0), mass=None, **kwargs):
    """
    Creates a inertial element with attributes specified by @**kwargs.

    Args:
        mass: The mass of inertial
    """
    if mass is not None:
        kwargs["mass"] = str(mass)
    kwargs["pos"] = array_to_string(pos)
    element = ET.Element("inertial", attrib=kwargs)
    return element


def get_size(size,
             size_max,
             size_min,
             default_max,
             default_min):
    """
    Helper method for providing a size, or a range to randomize from

    Args:
        size (n-array): Array of numbers that explicitly define the size
        size_max (n-array): Array of numbers that define the custom max size from which to randomly sample
        size_min (n-array): Array of numbers that define the custom min size from which to randomly sample
        default_max (n-array): Array of numbers that define the default max size from which to randomly sample
        default_min (n-array): Array of numbers that define the default min size from which to randomly sample

    Returns:
        np.array: size generated

    Raises:
        ValueError: [Inconsistent array sizes]
    """
    if len(default_max) != len(default_min):
        raise ValueError('default_max = {} and default_min = {}'
                         .format(str(default_max), str(default_min)) +
                         ' have different lengths')
    if size is not None:
        if (size_max is not None) or (size_min is not None):
            raise ValueError('size = {} overrides size_max = {}, size_min = {}'
                             .format(size, size_max, size_min))
    else:
        if size_max is None:
            size_max = default_max
        if size_min is None:
            size_min = default_min
        size = np.array([np.random.uniform(size_min[i], size_max[i])
                         for i in range(len(default_max))])
    return np.array(size)