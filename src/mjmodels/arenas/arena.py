import numpy as np
from src.mjmodels.base import MujocoXML
from src.utils.mjcf_utils import array_to_string, string_to_array

class Arena(MujocoXML):
    """Base arena class."""
    def set_origin(self, offset):
        """Applies a constant offset to all objects."""
        offset = np.array(offset)
        for node in self.worldbody.findall("./*[@pos]"):
            cur_pos = string_to_array(node.get("pos"))
            new_pos = cur_pos + offset
            node.set("pos", array_to_string(new_pos))