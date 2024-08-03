import sapien.core as sapien
from sapien.core import Articulation, Actor, Link
from sapien.core import RenderBody
import numpy as np
from typing import Union, List

from perception.core import _dense_sample_convex_pcd, get_pcd_normals_from_obj, get_pcd_from_actor
from scene.core import SpecifiedObject, TaskScene



class Drawer(SpecifiedObject):
    def __init__(self, storage_furniture_body: Articulation, drawer_body: Link, name:str=None):
        super().__init__()

        self.storage_furniture_body = storage_furniture_body
        self.drawer_body = drawer_body
        if name!=None:
            self.drawer_body.set_name(name)

    def get_name(self)->str:
        return self.drawer_body.get_name()

    def get_pose(self)->sapien.Pose:
        return self.drawer_body.get_pose()

    def get_pcd(self, dense_sample_convex:bool=False)->np.ndarray:   
        return get_pcd_from_actor(self.drawer_body, dense_sample_convex)

    def get_pcd_normals(self, dense_sample_convex:bool=False)->np.ndarray:
        return get_pcd_normals_from_obj(self.drawer_body, dense_sample_convex)

    def get_handle(self, handle_name:str=None)->RenderBody:
        if ~handle_name:
            # We assume that in the urdf file, the handle visual body is indexed the last 
            # And we assume only the graspable part of a handle could be called "handle" 
            return self.drawer_body.get_visual_bodies()[-1]
        else:
            for vis_body in self.drawer_body.get_visual_bodies():
                if vis_body.get_name()==handle_name:
                    return vis_body
                
    def get_open_distance(self)->float:
        open_distance_list = self.storage_furniture_body.get_qpos().tolist()
        id_open_distance_hash={}
        for joint in self.storage_furniture_body.get_active_joints():
            id_open_distance_hash[joint.get_child_link().id] = open_distance_list[0]
            open_distance_list.pop(0)

        return id_open_distance_hash[self.drawer_body.id]
    
    def load_in(self, task_scene: TaskScene):
        task_scene.object_list.append(self)



class StorageFurniture(SpecifiedObject):
    def __init__(self, storage_furniture_body: Articulation, name:str=None):
        super().__init__()

        self.storage_furniture_body = storage_furniture_body
        if name!=None:
            self.storage_furniture_body.set_name(name)

        self.handle_num=0
        self.handle_list: List[RenderBody]=[]
        self.handle_open_limits: List[float]=[]        
        self.drawer_list: List[Drawer]=[]

        for index, drawer_joint in enumerate(self.storage_furniture_body.get_active_joints()):
            # We assume that in the urdf file, the handle visual body is indexed the last 
            # And we assume only the graspable part of a handle could be called "handle" 
            handle: RenderBody = drawer_joint.get_child_link().get_visual_bodies()[-1]
            handle.set_name(f"{self.storage_furniture_body.get_name()}_handle_{index}")
            self.handle_num+=1
            self.handle_list.append(handle)
            self.handle_open_limits.append(drawer_joint.get_limits()[0][1])

            self.drawer_list.append(
                Drawer(
                    storage_furniture_body=self.storage_furniture_body, 
                    drawer_body=drawer_joint.get_child_link(), 
                    name=f"{self.storage_furniture_body.get_name()}_drawer_{index}",
                )
            )


    def get_name(self)->str: 
        return self.storage_furniture_body.get_name()
    

    def get_pose(self)->sapien.Pose:
        # Notice: the pose of StorageFurniture body may be different from the handle pose
        return self.storage_furniture_body.get_pose()
    

    def get_handle_by_name(self, handle_name: str)-> RenderBody:
        for handle in self.handle_list:
            if handle.get_name()==handle_name:
                return handle
        return None
            
    def get_drawer_by_name(self, drawer_name: str)->Drawer:
        for drawer in self.drawer_list:
            if drawer.get_name()==drawer_name:
                return drawer
        return None
    
    def get_handle_by_drawer(self, drawer: Drawer)->RenderBody:
        # We assume that in the urdf file, the handle visual body is indexed the last 
        # And we assume only the graspable part of a handle could be called "handle" 
        return drawer.drawer_body.get_visual_bodies()[-1]            

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
            
    
    def get_pcd(self, dense_sample_convex:bool=False)->np.ndarray: 
        # The storage furniture body pcd 
        storage_furniture_body_pcd=[]
        for link in self.storage_furniture_body.get_links():
            # The drawer should be isolatedly considered, because we need to get their open degree, 
            # and then get their real-time pointcloud, as they can be moving at any time.
            if link.get_name() in self.get_drawer_name_list():
                continue
            for vis_body in link.get_collision_visual_bodies():
                for render_shape in vis_body.get_render_shapes():
                    vertices = render_shape.mesh.vertices
                    if dense_sample_convex:
                        vertices=_dense_sample_convex_pcd(vertices)
                    storage_furniture_body_pcd+=vertices.tolist()
        storage_furniture_body_pcd=np.array(storage_furniture_body_pcd)
        # If the articulation.get_links()[1].get_visual_bodies() return [], just change the index 1 to other number
        storage_furniture_body_pcd=storage_furniture_body_pcd * self.storage_furniture_body.get_links()[1].get_visual_bodies()[0].scale

        tf_mat=self.storage_furniture_body.get_links()[1].get_pose().to_transformation_matrix()
        storage_furniture_body_pcd_homo=np.concatenate((storage_furniture_body_pcd, np.ones((storage_furniture_body_pcd.shape[0],1))), axis=-1)
        storage_furniture_body_pcd = (tf_mat@storage_furniture_body_pcd_homo.T).T[:,:-1]
        
        # The drawer pcd (including the handle) 
        drawer_pcd=[]
        for link in self.storage_furniture_body.get_links():
            # The drawer should be isolatedly considered, because we need to get their open degree, 
            # and then get their real-time pointcloud, as they can be moving at any time.
            if link.get_name() in self.get_drawer_name_list():
                drawer_pcd+=self.get_drawer_pcd_by_name(link.get_name(), dense_sample_convex).tolist()
        drawer_pcd=np.array(drawer_pcd)

        return np.concatenate((storage_furniture_body_pcd, drawer_pcd), axis=0)
    

    def get_pcd_normals(self, dense_sample_convex:bool=False)->np.ndarray:
        return get_pcd_normals_from_obj(self.storage_furniture_body, dense_sample_convex)


    def get_handle_pcd_by_name(self, handle_name: str, dense_sample_convex:bool=False)->np.ndarray:
        handle : RenderBody = self.get_handle_by_name(handle_name)
        
        handle_pcd = handle.get_render_shapes()[0].mesh.vertices
        handle_pcd = handle_pcd * handle.scale

        # tf_mat = self.storage_furniture_body.get_pose().to_transformation_matrix()
        handle_tf_mat = self.storage_furniture_body.get_active_joints()[0].get_parent_link().get_pose().to_transformation_matrix()
        handle_pcd_homo = np.concatenate((handle_pcd, np.ones((handle_pcd.shape[0], 1))), axis=-1)
        handle_pcd=(handle_tf_mat@handle_pcd_homo.T).T[:,:-1]

        # We assume that the initial (when the urdf is loaded in) forward direction of the handle is [-1,0,0]
        storage_furniture_tf_mat = self.storage_furniture_body.get_pose().to_transformation_matrix()
        open_direction = (storage_furniture_tf_mat[:3,:3]@np.array([-1., 0., 0.]).T).T
        open_direction = open_direction/np.linalg.norm(open_direction)

        open_distance = self.get_open_distance_by_name(handle_name)
        handle_pcd += open_distance * open_direction

        if dense_sample_convex:
            handle_pcd=_dense_sample_convex_pcd(handle_pcd)

        return handle_pcd
    

    def get_drawer_pcd_by_name(self, drawer_name: str, dense_sample_convex:bool=False)->np.ndarray:
        drawer_body : Link = self.get_drawer_by_name(drawer_name).drawer_body  
        drawer_pcd = get_pcd_from_actor(drawer_body, dense_sample_convex)

        # # We assume that the initial (when the urdf is loaded in) forward direction of the handle is [-1,0,0]
        # storage_furniture_tf_mat = self.storage_furniture_body.get_pose().to_transformation_matrix()
        # open_direction = (storage_furniture_tf_mat[:3,:3]@np.array([-1., 0., 0.]).T).T
        # open_direction = open_direction/np.linalg.norm(open_direction)

        # open_distance = self.get_open_distance_by_name(self.get_handle_by_drawer(drawer).get_name())
        # drawer_pcd += open_distance * open_direction

        return drawer_pcd


    def get_open_degree_by_name(self, handle_name:str)->float:
        return self.get_open_distance_by_name(handle_name)/self.get_open_limit_by_name(handle_name)
    

    def get_open_distance_by_name(self, handle_name:str)->float:
        open_distance_list = self.storage_furniture_body.get_qpos().tolist()
        id_open_distance_hash={}
        for joint in self.storage_furniture_body.get_active_joints():
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
        new_qpos = self.storage_furniture_body.get_qpos().tolist()        
        new_qpos[handle_index] = open_limit * degree 
        self.storage_furniture_body.set_qpos(new_qpos)

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
            name:str=None
        ):
        super().__init__()

        self.task_scene=task_scene
        # The robot_articulation may be different from the robot articulation loaded from the urdf file
        # Because the robot articulation may have been mounted with some camera entities, 
        # the camera entities are "links", but the robot articulation loaded from the urdf file doesn't contain them
        self.robot_articulation=robot_articulation
        self.urdf_file_path=urdf_file_path
        self.srdf_file_path=srdf_file_path
        self.move_group=move_group
        self.active_joints_num_wo_MG=active_joints_num_wo_EE
        self.joint_vel_limits=np.ones(active_joints_num_wo_EE)#*0.2
        self.joint_acc_limits=np.ones(active_joints_num_wo_EE)#*0.2
        self.mounted_obj=mounted_obj

        self.origin_link_names, self.origin_joint_names = self._get_origin_link_joint()
        
        if name!=None:
            self.robot_articulation.set_name(name)
        
    def get_name(self):
        return self.robot_articulation.get_name()

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