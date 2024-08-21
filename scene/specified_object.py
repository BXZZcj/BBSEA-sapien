import sapien.core as sapien
from sapien.core import Articulation, Actor, Link
from sapien.core import RenderBody
import numpy as np
from typing import Union, List

from perception.core import \
    get_pcd_from_obj, \
    get_pcd_normals_from_obj, \
    _get_pcd_from_vis_body
from scene.core import SpecifiedObject, TaskScene



class StorageFurnitureDrawer(SpecifiedObject):
    def __init__(
            self, 
            storage_furniture_body: Articulation, 
            drawer_body: Link, 
            name:str=None,
            parent: SpecifiedObject = None,
    ):
        super().__init__()

        self.storage_furniture_body = storage_furniture_body
        self.body = drawer_body
        if name!=None:
            self.body.set_name(name)

        self.parent = parent


    def get_handle(self, handle_name:str=None)->RenderBody:
        if ~handle_name:
            # We assume that in the urdf file, the handle visual body is indexed the last 
            # And we assume only the graspable part of a handle could be called "handle" 
            return self.body.get_visual_bodies()[-1]
        else:
            for vis_body in self.body.get_visual_bodies():
                if vis_body.get_name()==handle_name:
                    return vis_body
                
    def get_open_distance(self)->float:
        open_distance_list = self.storage_furniture_body.get_qpos().tolist()
        id_open_distance_hash={}
        for joint in self.storage_furniture_body.get_active_joints():
            id_open_distance_hash[joint.get_child_link().id] = open_distance_list[0]
            open_distance_list.pop(0)

        return id_open_distance_hash[self.body.id]



class StorageFurniture(SpecifiedObject):
    def __init__(
            self, 
            storage_furniture_body: Articulation, 
            name:str=None,
            parent: SpecifiedObject = None,
    ):
        super().__init__()

        self.body = storage_furniture_body
        if name != None:
            self.body.set_name(name)
        self.parent = parent

        self.handle_num = 0
        self.handle_list: List[RenderBody]=[]
        self.handle_open_limits: List[float]=[]        
        self.drawer_list: List[StorageFurnitureDrawer]=[]

        for index, drawer_joint in enumerate(self.body.get_active_joints()):
            # We assume that in the urdf file, the handle visual body is indexed the last 
            # And we assume only the graspable part of a handle could be called "handle" 
            handle: RenderBody = drawer_joint.get_child_link().get_visual_bodies()[-1]
            handle.set_name(f"{self.body.get_name()}_handle_{index}")
            self.handle_num += 1
            self.handle_list.append(handle)
            self.handle_open_limits.append(drawer_joint.get_limits()[0][1])

            self.drawer_list.append(
                StorageFurnitureDrawer(
                    storage_furniture_body=self.body, 
                    drawer_body=drawer_joint.get_child_link(), 
                    name=f"{self.body.get_name()}_drawer_{index}",
                    parent=self,
                )
            )


    def get_handle_by_name(self, handle_name: str)-> RenderBody:
        for handle in self.handle_list:
            if handle.get_name()==handle_name:
                return handle
        return None
            
    def get_drawer_by_name(self, drawer_name: str)->StorageFurnitureDrawer:
        for drawer in self.drawer_list:
            if drawer.get_name()==drawer_name:
                return drawer
        return None
    
    def get_handle_by_drawer(self, drawer: StorageFurnitureDrawer)->RenderBody:
        # We assume that in the urdf file, the handle visual body is indexed the last 
        # And we assume only the graspable part of a handle could be called "handle" 
        return drawer.body.get_visual_bodies()[-1]            

    def get_handle_name_list(self)->List[str]:
        handle_name_list=[]
        for handle in self.handle_list:
            handle_name_list.append(handle.get_name())
        return handle_name_list
    

    def get_drawer_name_list(self)->List[str]:
        drawer_name_list=[]
        for drawer in self.drawer_list:
            drawer_name_list.append(drawer.get_name())
        return drawer_name_list
            
            
    def get_handle_pcd_by_name(self, handle_name: str, dense_sample_convex:bool=False)->np.ndarray:
        handle_vis_body : RenderBody = self.get_handle_by_name(handle_name)
        
        drawer_link = self.get_drawer_by_name(handle_name.replace("handle", "drawer")).body
        for vis_body_index, vis_body in enumerate(drawer_link.get_visual_bodies()):
            if vis_body.get_visual_id() == handle_vis_body.get_visual_id():
                handle_vis_body_index = vis_body_index
                break
        return _get_pcd_from_vis_body(drawer_link, handle_vis_body, handle_vis_body_index, dense_sample_convex)
    

    def get_drawer_pcd_by_name(self, drawer_name: str)->np.ndarray:
        drawer_body : Link = self.get_drawer_by_name(drawer_name).body  
        return get_pcd_from_obj(drawer_body)


    def get_open_degree_by_name(self, handle_name:str)->float:
        return self.get_open_distance_by_name(handle_name)/self.get_open_limit_by_name(handle_name)
    

    def get_open_distance_by_name(self, handle_name:str)->float:
        open_distance_list = self.body.get_qpos().tolist()
        id_open_distance_hash={}
        for joint in self.body.get_active_joints():
            id_open_distance_hash[joint.get_child_link().id] = open_distance_list[0]
            open_distance_list.pop(0)

        handle_id = self.get_handle_by_name(handle_name).get_actor_id()
        return id_open_distance_hash[handle_id]
    

    def get_open_limit_by_name(self, handle_name: str)->float:
        for handle, open_limit in zip(self.handle_list, self.handle_open_limits):
            if handle.get_name()==handle_name:
                return open_limit
        return None
    

    def set_open_degree_by_name(self, handle_name: str, degree: float):
        assert degree>=0 and degree<=1, "The degree should be between 0 and 1"

        handle_index = int(handle_name.split("_")[-1])
        open_limit = self.get_open_limit_by_name(handle_name)
        new_qpos = self.body.get_qpos().tolist()        
        new_qpos[handle_index] = open_limit * degree 
        self.body.set_qpos(new_qpos)


    def load_in(self, task_scene:TaskScene):
        task_scene.object_list.append(self)
        for drawer in self.drawer_list:
            drawer.load_in(task_scene=task_scene)
    


class Robot(SpecifiedObject):
    def __init__(
            self,
            task_scene: TaskScene,
            robot_articulation: Articulation, 
            urdf_file_path:str,
            srdf_file_path:str,
            move_group:str,
            active_joints_num_wo_EE:int,
            mounted_obj:list[Link]=[],
            name:str=None,
            parent: SpecifiedObject = None,
        ):
        super().__init__()

        self.task_scene=task_scene
        # The robot_articulation may be different from the robot articulation loaded from the urdf file
        # Because the robot articulation may have been mounted with some camera entities, 
        # the camera entities are "links", but the robot articulation loaded from the urdf file doesn't contain them
        self.body=robot_articulation
        self.urdf_file_path=urdf_file_path
        self.srdf_file_path=srdf_file_path
        self.move_group=move_group
        self.active_joints_num_wo_MG=active_joints_num_wo_EE
        self.joint_vel_limits=np.ones(active_joints_num_wo_EE)#*0.2
        self.joint_acc_limits=np.ones(active_joints_num_wo_EE)#*0.2
        self.mounted_obj=mounted_obj

        self.origin_link_names, self.origin_joint_names = self._get_origin_link_joint()
        
        if name!=None:
            self.body.set_name(name)

        self.parent = parent

    def _get_origin_link_joint(self):
        robot_loader = self.task_scene.scene.create_urdf_loader()
        origin_robot_articulation = robot_loader.load(self.urdf_file_path)
        robot_loader.fix_root_link=True
        origin_link_names = [link.get_name() for link in origin_robot_articulation.get_links()]
        origin_joint_names = [joint.get_name() for joint in origin_robot_articulation.get_active_joints()]
        self.task_scene.scene.remove_articulation(origin_robot_articulation)
        return origin_link_names, origin_joint_names


    def set_joint_vel_limits(self, joint_vel_limits:np.ndarray):
        assert len(joint_vel_limits)==self.active_joints_num_wo_MG, \
            "The length of joint_vel_limits should be the same as the active joints number except the move group."
        self.joint_vel_limits=joint_vel_limits


    def set_joint_acc_limits(self, joint_acc_limits:np.ndarray):
        assert len(joint_acc_limits)==self.active_joints_num_wo_MG, \
            "The length of joint_acc_limits should be the same as the active joints number except the move group."
        self.joint_acc_limits=joint_acc_limits

    
    def add_mounted_obj(self, obj:Link):
        self.mounted_obj.append(obj)


    def get_mounted_obj(self):
        return self.mounted_obj
    
    def load_in(self, task_scene: TaskScene):
        task_scene.robot_list.append(self)



class CatapultArm(SpecifiedObject):
    def __init__(
            self, 
            catapult_body: Articulation,
            catapult_arm_body: Link,
            catapult_arm_name: str=None,
            parent: SpecifiedObject = None,
    ):
        self.catapult_body=catapult_body
        self.body=catapult_arm_body
        
        if catapult_arm_name != None:
            self.body.set_name(catapult_arm_name)

        self.parent = parent



class CatapultButton(SpecifiedObject):
    def __init__(
            self,
            catapult_body: Articulation,
            button_body: Link,
            button_name:str = None,
            parent: SpecifiedObject = None,
    ):
        self.catapult_body = catapult_body
        self.body = button_body

        if button_name != None:
            self.body.set_name(button_name)

        self.parent = parent

        assert self.body.get_name()!=None, "The button name must be set."

        for joint in self.catapult_body.get_active_joints():
            if joint.get_child_link().get_name()==self.body.get_name():
                self.button_joint = joint
                break


    def get_press_limit(self)->float:
        button_joint_limit = self.button_joint.get_limits()
        return button_joint_limit.max()-button_joint_limit.min()


    def get_press_distance(self)->float:
        for joint_index, joint in enumerate(self.catapult_body.get_active_joints()):
            if joint.get_name()==self.button_joint.get_name():
                button_joint_index=joint_index
        return self.catapult_body.get_qpos()[button_joint_index]



class Catapult(SpecifiedObject):
    def __init__(
            self,
            catapult_body: Articulation, 
            catapult_name:str=None,
            initial_qf:np.ndarray=None,
            parent: SpecifiedObject = None,
    ):
        self.body=catapult_body

        if catapult_name!=None:
            self.body.set_name(catapult_name)

        assert self.body.get_name()!=None, "The catapult name must be set."

        for link in self.body.get_links():
            if link.get_name() == "catapult_arm":
                link.set_name(self.body.get_name()+"_arm")
                self.catapult_arm = CatapultArm(self.body, link, self.body.get_name()+"_arm", parent=self)
            if link.get_name() == "button":
                link.set_name(self.body.get_name()+"_button")
                self.button = CatapultButton(self.body, link, self.body.get_name()+"_button", parent=self)
        
        for joint_index, joint in enumerate(self.body.get_active_joints()):
            if joint.get_child_link().get_name()=="catapult_arm":
                # We assume the dof of button is only 1, so the index can be fixed 0.
                self.catapult_arm_joint = joint
                self.catapult_arm_joint_index = joint_index
            if joint.get_child_link().get_name()=="button":
                # We assume the dof of button is only 1, so the index can be fixed 0.
                self.button_joint = joint
                self.button_joint_index = joint_index
        
        self.initial_qf=initial_qf
        if self.initial_qf is not None:
            self.body.set_qf(self.initial_qf)

        self.parent = parent


    def check_activate(self):
        return np.isclose(self.button.get_press_limit()-self.button.get_press_distance(), 0, atol=1e-3)
    

    def activate_behavior(self):
        current_qf = self.body.get_qf()
        current_qf[self.catapult_arm_joint_index] = 20
        self.body.set_qf(current_qf)

    
    def reset(self):
        self.body.set_qf(self.initial_qf)

    
    def load_in(self, task_scene: TaskScene):
        task_scene.object_list.append(self)
        task_scene.object_list.append(self.catapult_arm)
        task_scene.object_list.append(self.button)