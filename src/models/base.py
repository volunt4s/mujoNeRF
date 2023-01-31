import os
import xml.etree.ElementTree as ET
import io

OBJECT_TYPE = ["", "object", "collision", "visual"]

class MujocoXML:
    """
    Base class of Mujoco xml file
    Wraps around ElementTree and provides additional functionality for merging different models.
    Specially, we keep track of <worldbody/>, <actuator/> and <asset/>
    """

    def __init__(self, fname):
        """
        Loads a mujoco xml from file.
        Args:
            fname (str): path to the MJCF xml file.
        """
        self.file = fname
        self.folder = os.path.dirname(fname)
        self.tree = ET.parse(fname)
        self.root = self.tree.getroot()
        self.name = self.root.get("model")
        self.worldbody = self.create_default_element("worldbody")
        self.actuator = self.create_default_element("actuator")
        self.asset = self.create_default_element("asset")
        self.equality = self.create_default_element("equality")
        self.sensor = self.create_default_element("sensor")
        self.contact = self.create_default_element("contact")
        self.default = self.create_default_element("default")
        self.tendon = self.create_default_element("tendon")
        self.resolve_asset_dependency()

    def resolve_asset_dependency(self):
        """
        Converts every file dependency into absolute path so when we merge we don't break things.
        """

        for node in self.asset.findall("./*[@file]"):
            file = node.get("file")
            abs_path = os.path.abspath(self.folder)
            abs_path = os.path.join(abs_path, file)
            node.set("file", abs_path)

    def create_default_element(self, name):
        """
        Creates a <@name/> tag under root if there is none.
        """

        found = self.root.find(name)
        if found is not None:
            return found
        ele = ET.Element(name)
        self.root.append(ele)
        return ele

    def merge(self, other, merge_body=True, object_type=""):
        """
        Default merge method.
        Args:
            other: another MujocoXML instance
                raises XML error if @other is not a MujocoXML instance.
                merges <worldbody/>, <actuator/> and <asset/> of @other into @self
            merge_body: True if merging child bodies of @other. Defaults to True.
        """
        if object_type not in OBJECT_TYPE:
            raise NameError(f"Check the body name, body name is one of {OBJECT_TYPE}")

        self.merge_asset(other)
        if object_type == "":
            if merge_body:
                for body in other.worldbody:
                    self.worldbody.append(body)
            
            for one_actuator in other.actuator:
                self.actuator.append(one_actuator)
            for one_equality in other.equality:
                self.equality.append(one_equality)
            for one_sensor in other.sensor:
                self.sensor.append(one_sensor)
            for one_contact in other.contact:
                self.contact.append(one_contact)
            for one_default in other.default:
                self.default.append(one_default)
            for one_tendon in other.tendon:
                self.tendon.append(one_tendon)
            
        if object_type == "object":
            obj = other.get_collision()
            self.worldbody.append(obj)

    def merge_asset(self, other):
        """
        Merges other files in a custom logic.
        Args:
            other (MujocoXML): other xml file whose assets will be merged into this one
        """
        for asset in other.asset:
            asset_name = asset.get("name")
            asset_type = asset.tag
            # Avoids duplication
            pattern = "./{}[@name='{}']".format(asset_type, asset_name)
            if self.asset.find(pattern) is None:
                self.asset.append(asset)

    def merge_camera_set(self, camera_set_xml):
        for camera in camera_set_xml:
            self.worldbody.append(camera)

    def get_model(self, mode="mujoco"):
        """
        Returns a MjModel instance from the current xml tree.
        """
        available_modes = ["dm_control", "mujoco"]
        xml_string = self.get_xml()
        if mode == "dm_control":
            from dm_control import mujoco
            model = mujoco.Physics.from_xml_string(xml_string)
            return model
        elif mode == "mujoco":
            import mujoco
            model = mujoco.MjModel.from_xml_string(xml_string)
            return model
        raise ValueError(
                "Unkown model mode: {}. Available options are: {}".format(
                    mode, ",".join(available_modes)
                )
        )

    def get_xml(self):
        """
        Returns a string of the MJCF XML file.
        """
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            return string.getvalue()