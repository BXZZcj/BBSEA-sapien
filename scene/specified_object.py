from sapien.core import Articulation, Actor
from sapien.core import RenderBody
import sapien.core as sapien
import numpy as np

from perception.core import get_pcd_from_articulation, get_pcd_normals_from_obj, get_pcd_from_actor
from scene.core import SpecifiedObject


class StorageFurniture(SpecifiedObject):
    def __init__(self, storage_furniture_body: Articulation, name:str=None):
        super().__init__()

        self.storage_furniture_body = storage_furniture_body
        if name!=None:
            self.storage_furniture_body.set_name(name)

        self.handle_num=0
        self.handle_list=[]
        self.handle_open_limits=[]        
        self.drawer_list=[]

        for index, drawer_joint in enumerate(self.storage_furniture_body.get_active_joints()):
            # We assume that in the urdf file, the handle visual body is indexed the last 
            # And we assume only the graspable part of a handle could be called "handle" 
            handle = drawer_joint.get_child_link().get_visual_bodies()[-1]
            handle.set_name(f"{self.storage_furniture_body.get_name()}_handle_{index}")
            self.handle_num+=1
            self.handle_list.append(handle)
            self.handle_open_limits.append(drawer_joint.get_limits()[0][1])

            drawer = drawer_joint.get_child_link()
            drawer.set_name(f"{self.storage_furniture_body.get_name()}_drawer_{index}")
            self.drawer_list.append(drawer)


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
            
    def get_drawer_by_name(self, drawer_name: str)->Actor:
        for drawer in self.drawer_list:
            if drawer.get_name()==drawer_name:
                return drawer
        return None
    
    def get_handle_by_drawer(self, drawer: Actor)->RenderBody:
        return self.get_handle_by_name(drawer.get_visual_bodies()[-1].get_name())            

    def get_handle_name_list(self)->list:
        handle_name_list=[]
        for handle in self.handle_list:
            handle_name_list.append(handle.get_name())
        return handle_name_list
    

    def get_drawer_name_list(self)->list:
        drawer_name_list=[]
        for drawer in self.drawer_list:
            drawer_name_list.append(drawer.get_name())
        return drawer_name_list
            
    
    def get_pcd(self)->np.ndarray: 
        return get_pcd_from_articulation(self.storage_furniture_body)
    
    def get_pcd_normals(self)->np.ndarray:
        return get_pcd_normals_from_obj(self.storage_furniture_body)


    def get_handle_pcd_by_name(self, handle_name: str)->np.ndarray:
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

        return handle_pcd
    

    def get_drawer_pcd_by_name(self, drawer_name: str)->np.ndarray:
        drawer : Actor = self.get_drawer_by_name(drawer_name)        
        drawer_pcd = get_pcd_from_actor(drawer)

        # We assume that the initial (when the urdf is loaded in) forward direction of the handle is [-1,0,0]
        storage_furniture_tf_mat = self.storage_furniture_body.get_pose().to_transformation_matrix()
        open_direction = (storage_furniture_tf_mat[:3,:3]@np.array([-1., 0., 0.]).T).T
        open_direction = open_direction/np.linalg.norm(open_direction)

        open_distance = self.get_open_distance_by_name(self.get_handle_by_drawer(drawer).get_name())
        drawer_pcd += open_distance * open_direction

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

        handle_index = handle_name.split("_")[-1]
        open_limit = self.get_open_limit_by_name(handle_name)
        new_qpos = self.get_open_distance_by_name(handle_name)
        new_qpos[handle_index] = open_limit[handle_index] * degree 
        self.storage_furniture_body.set_qpos(new_qpos)


class Drawer(SpecifiedObject):
    def __init__(self, storage_furniture_body: Articulation, drawer: Actor, name:str=None):
        super().__init__()

        self.storage_furniture_body = storage_furniture_body
        self.drawer = drawer
        if name!=None:
            self.drawer.set_name(name)

    def get_name(self):
        return self.drawer.get_name()

    def get_pose(self)->sapien.Pose:
        return self.drawer.get_pose()

    def get_pcd(self)->np.ndarray:   
        drawer_pcd = get_pcd_from_actor(self.drawer)

        # We assume that the initial (when the urdf is loaded in) forward direction of the handle is [-1,0,0]
        storage_furniture_tf_mat = self.storage_furniture_body.get_pose().to_transformation_matrix()
        open_direction = (storage_furniture_tf_mat[:3,:3]@np.array([-1., 0., 0.]).T).T
        open_direction = open_direction/np.linalg.norm(open_direction)

        open_distance = self.get_open_distance()
        drawer_pcd += open_distance * open_direction

        return drawer_pcd

    def get_pcd_normals(self)->np.ndarray:
        pass

    def get_handle(self, handle_name:str=None)->RenderBody:
        if ~handle_name:
            # We assume that in the urdf file, the handle visual body is indexed the last 
            # And we assume only the graspable part of a handle could be called "handle" 
            return self.drawer.get_visual_bodies()[-1]
        else:
            for vis_body in self.drawer.get_visual_bodies():
                if vis_body.get_name()==handle_name:
                    return vis_body
                
    def get_open_distance(self)->float:
        open_distance_list = self.storage_furniture_body.get_qpos().tolist()
        id_open_distance_hash={}
        for joint in self.storage_furniture_body.get_active_joints():
            id_open_distance_hash[joint.get_child_link().id] = open_distance_list[0]
            open_distance_list.pop(0)

        return id_open_distance_hash[self.drawer.id]