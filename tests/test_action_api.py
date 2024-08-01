import sys
import sapien.core as sapien
import numpy as np
from transforms3d.euler import euler2quat
import transforms3d.quaternions as quat 

from manipulate.scene.SimplePickPlace_v0 import SimplePickPlaceScene

demo=SimplePickPlaceScene()
demo.scene.step() 
demo.scene.update_render()

# get_names_on_table=demo.get_names_on_table
# get_location_by_name=demo.get_pose_by_name
# get_box_position=demo.get_pad_postion
# move_tool=demo.grasp_tool.move_for_grasp
# grasp=demo.grasp_tool.execute
# ungrasp=demo.grasp_tool.ungrasp

# def plan():
#     box_position = get_box_position()
#     object_names = get_names_on_table()
#     for name in object_names:
#         object_position = get_location_by_name(name)
#         move_tool(object_position)
#         grasp()
#         move_tool(box_position)
#         ungrasp()


# x,y,z=1,1,0
# rotation_matrix = np.array([
#     [0, y, x],
#     [0, -x, y],
#     [1, 0, z]
# ])

# success=demo.grasp_tool._move_to_pose(sapien.Pose([0.5,0,0.6], euler2quat(np.pi,0,0)))

# demo.primitives.Push("sphere", [1,0,0],0.1)
demo.primitives.Push("box",[0,-1],0.1)
demo.primitives.Pick("banana")
demo.primitives.PlaceOn("green_pad")
# demo.primitives._open_gripper()
# demo.primitives.PlaceAt([0.56, 0.35, 0.11])
# demo.primitives.PrismaticJointOpen("StorageFurniture45290_handle_2")
# plan()
# while not demo.viewer.closed:
#     demo.scene.update_render()
#     demo.viewer.render()
# demo.grasp_tool._close_gripper()
# demo.grasp_tool._move_to_pose(sapien.Pose([0.9,0.9,0.9], [0,1,0,0]))