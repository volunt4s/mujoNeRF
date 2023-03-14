from src.mjmodels.base import MujocoXML
from src.utils.mjcf_utils import xml_path_completion

class MujocoWorldBase(MujocoXML):
    def __init__(self):
        super().__init__(xml_path_completion('base.xml'))