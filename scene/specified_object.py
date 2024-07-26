from sapien.core import Articulation
from sapien.core import RenderBody
import sapien.core as sapien
import numpy as np

from perception.core import get_pcd_from_articulation, get_pcd_normals_from_articulation
from scene.core import SpecifiedObject


class Drawer(SpecifiedObject):
    def __init__(self, drawer_body: Articulation, name=None):
        super().__init__()

        self.drawer_body = drawer_body
        if name!=None:
            self.drawer_body.set_name(name)

        self.handle_num=0
        self.handle_list=[]
        self.handle_open_limits=[]

        for index, tier in enumerate(self.drawer_body.get_active_joints()):
            # We assume that in the urdf file, the handle visual body is indexed the last 
            handle = tier.get_child_link().get_visual_bodies()[-1]
            handle.set_name(f"{self.drawer_body.get_name()}_handle_{index}")
            self.handle_num+=1
            self.handle_list.append(handle)
            self.handle_open_limits.append(tier.get_limits()[0][1])


    def get_name(self)->str: 
        return self.drawer_body.get_name()
    

    def get_pose(self)->sapien.Pose:
        # Notice: the pose of drawer body may be different from the handle pose
        return self.drawer_body.get_pose()
    

    def get_handle_by_name(self, obj_name: str)-> RenderBody:
        for handle in self.handle_list:
            if handle.get_name()==obj_name:
                return handle
            

    def get_handle_name_list(self)->list:
        handle_name_list=[]
        for handle in self.handle_list:
            handle_name_list.append(handle.get_name())
        return handle_name_list
            
    
    def get_pcd(self)->np.ndarray: 
        return get_pcd_from_articulation(self.drawer_body)
    
    def get_pcd_normals(self)->np.ndarray:
        return get_pcd_normals_from_articulation(self.drawer_body)


    def get_handle_pcd_by_name(self, handle_name: str)->np.ndarray:
        handle = self.get_handle_by_name(handle_name)
        
        handle_pcd = handle.get_render_shapes()[0].mesh.vertices
        handle_pcd = handle_pcd * handle.scale

        # tf_mat = self.drawer_body.get_pose().to_transformation_matrix()
        handle_tf_mat = self.drawer_body.get_active_joints()[0].get_parent_link().get_pose().to_transformation_matrix()
        handle_pcd_homo = np.concatenate((handle_pcd, np.ones((handle_pcd.shape[0], 1))), axis=-1)
        handle_pcd=(handle_tf_mat@handle_pcd_homo.T).T[:,:-1]

        # We assume that the initial (when the urdf is loaded in) forward direction of the handle is [-1,0,0]
        drawer_tf_mat = self.drawer_body.get_pose().to_transformation_matrix()
        open_direction = (drawer_tf_mat[:3,:3]@np.array([-1., 0., 0.]).T).T
        open_direction = open_direction/np.linalg.norm(open_direction)
        def _get_open_upper_limit_by_name(handle_name:str)->float:
            for index, handle in enumerate(self.handle_list):
                if handle.get_name() == handle_name:
                    return self.handle_open_limits[index]
        open_distance = self.get_open_degree_by_name(handle_name) * _get_open_upper_limit_by_name(handle_name)
        handle_pcd += open_distance * open_direction

        return handle_pcd
    

    def get_open_degree_by_name(self, handle_name:str)->float:
        open_degree_list = (self.drawer_body.get_qpos()/self.drawer_body.get_qlimits()[:,-1]).tolist()
        id_open_degree_hash={}
        for joint in self.drawer_body.get_active_joints():
            id_open_degree_hash[joint.get_child_link().id] = open_degree_list[0]
            open_degree_list.pop(0)

        handle_id = self.get_handle_by_name(handle_name).get_actor_id()
        return id_open_degree_hash[handle_id]
    

    def get_open_limit_by_name(self, handle_name: str)->float:
        for handle, open_limit in zip(self.handle_list, self.handle_open_limits):
            if handle.get_name()==handle_name:
                return open_limit
        return None