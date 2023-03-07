from src.models.objects import MujocoXMLObject
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

class BottleObject(MujocoXMLObject):
    """
    Bottle object
    """

    def __init__(
        self, 
        fname="objects/bottle.xml",
        name="bottle",
        joints=[dict(type="free", damping="0.0005")],
        pos=[0, 0, 0],
        rot=[0, 0, 0],
    ):
        super().__init__(fname=xml_path_completion(fname), 
                         name=name, joints=joints, pos=pos, rot=rot)


class CanObject(MujocoXMLObject):
    """
    Can object
    """

    def __init__(
        self, 
        fname="objects/can.xml",
        name="can",
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


class LemonObject(MujocoXMLObject):
    """
    Lemon object
    """

    def __init__(
        self, 
        fname="objects/lemon.xml",
        name="lemon",
        joints=[dict(type="free", damping="0.0005")],
        pos=[0, 0, 0],
        rot=[0, 0, 0],
    ):
        super().__init__(fname=xml_path_completion(fname), 
                         name=name, joints=joints, pos=pos, rot=rot)


class MilkObject(MujocoXMLObject):
    """
    Milk object
    """

    def __init__(
        self, 
        fname="objects/milk.xml",
        name="milk",
        joints=[dict(type="free", damping="0.0005")],
        pos=[0, 0, 0],
        rot=[0, 0, 0],
    ):
        super().__init__(fname=xml_path_completion(fname), 
                         name=name, joints=joints, pos=pos, rot=rot)


class BreadObject(MujocoXMLObject):
    """
    Bread object
    """

    def __init__(
        self, 
        fname="objects/bread.xml",
        name="bread",
        joints=[dict(type="free", damping="0.0005")],
        pos=[0, 0, 0],
        rot=[0, 0, 0],
    ):
        super().__init__(fname=xml_path_completion(fname), 
                         name=name, joints=joints, pos=pos, rot=rot)


class CerealObject(MujocoXMLObject):
    """
    Cereal object
    """

    def __init__(
        self, 
        fname="objects/cereal.xml",
        name="cereal",
        joints=[dict(type="free", damping="0.0005")],
        pos=[0, 0, 0],
        rot=[0, 0, 0],
    ):
        super().__init__(fname=xml_path_completion(fname), 
                         name=name, joints=joints, pos=pos, rot=rot)


class DoorObject(MujocoXMLObject):
    """
    Door object
    """

    def __init__(
        self, 
        fname="objects/door.xml",
        name="door",
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
