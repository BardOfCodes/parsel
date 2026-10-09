import numpy as np
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

BLK_SIZE = 128
W = BLK_SIZE * 12
H = BLK_SIZE * 9
GIF_W = BLK_SIZE * 4
GIF_H = BLK_SIZE * 4

GUI_COLOR = gui.Color(0.2, 0.5, 0.2, 1.0)
RED_COLOR = gui.Color(1.0, 0.0, 0.0, 1.0)
GUI_COLOR_2 = gui.Color(0.2, 0.0, 0.8, 1.0)

LIT_MATERIAL = rendering.MaterialRecord()
LIT_MATERIAL.shader = "defaultLit"
UNLIT_MATERIAL = LIT_MATERIAL
PC_MATERIAL = rendering.MaterialRecord()
PC_MATERIAL.point_size = 50.0

MESH_ITEM = 0
OBB_ITEM = 1
LABEL_ITEM = 2
PC_ITEM = 3
AXIS_ITEM = 4
COLORS = np.random.default_rng(42).uniform(0, 1, size=(1000, 3))
