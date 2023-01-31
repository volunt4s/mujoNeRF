import numpy as np
from src.models.arenas import Arena
from src.utils.mjcf_utils import xml_path_completion
from src.utils.mjcf_utils import array_to_string, string_to_array


class TableArena(Arena):
    """Workspace that contains a tabletop."""

    def __init__(self, table_full_size=(0.35, 0.6, 0.02), table_friction=(1, 0.005, 0.0001)):
        """
        Args:
            table_full_size: full dimensions of the table
            table_friction: friction parameters of the table
        """
        super().__init__(xml_path_completion("arenas/table_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.floor = self.worldbody.find("./geom[@name='floor']")
        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_view = self.table_body.find("./geom[@name='table_view']")
        self.configure_location()

    def configure_location(self):
        self.bottom_pos = np.array([0, 0, 0])
        self.floor.set("pos", array_to_string(self.bottom_pos))
        self.table_collision.set("size", array_to_string(self.table_full_size))
        self.table_view.set("size", array_to_string(self.table_full_size))

    @property
    def table_top_abs(self):
        """
        Returns the absolute position of table top.
        """
        return string_to_array(self.table_body.get("pos"))

