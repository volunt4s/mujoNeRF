from src.mjmodels.objects import MujocoXMLObject
from src.utils.mjcf_utils import xml_path_completion


class CustomObject(MujocoXMLObject):
    """
    Custom object
    """

    def __init__(
        self, 
        fname="objects/square_box.xml",
        name='custom',
        joints=[dict(type="free", damping="0.0005")],
        pos=[0, 0, 0],
        rot=[0, 0, 0],
    ):
        super().__init__(fname=xml_path_completion(fname), 
                         name=name, joints=joints, pos=pos, rot=rot)


class CatObject(MujocoXMLObject):
    """
    Can object
    """

    def __init__(
        self, 
        fname="objects/cat.xml",
        name="cat",
        joints=[dict(type="free", damping="0.0005")],
        pos=[0, 0, 0],
        rot=[0, 0, 0],
    ):
        super().__init__(fname=xml_path_completion(fname), 
                         name=name, joints=joints, pos=pos, rot=rot)


class LegoObject(MujocoXMLObject):
    def __init__(
        self, 
        fname="objects/lego.xml",
        name="lego",
        joints=[dict(type="free", damping="0.0005")],
        pos=[0, 0, 0],
        rot=[0, 0, 0],
    ):
        super().__init__(fname=xml_path_completion(fname), 
                         name=name, joints=joints, pos=pos, rot=rot)


class DragonObject(MujocoXMLObject):
    def __init__(
        self, 
        fname="objects/dragon.xml",
        name="dragon",
        joints=[dict(type="free", damping="0.0005")],
        pos=[0, 0, 0],
        rot=[0, 0, 0],
    ):
        super().__init__(fname=xml_path_completion(fname), 
                         name=name, joints=joints, pos=pos, rot=rot)