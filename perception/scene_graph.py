import torch
import numpy as np
import scipy
import cv2
import open3d as o3d
from torchvision.io import read_image
from PIL import Image

from config import *
from perception.clip_utils import CLIP_classify

# =========  Parameters for spatial relation heuristics ============
IN_CONTACT_DISTANCE = 0.01
INSIDE_THRESH = 0.5
# ON_TOP_OF_THRESH = 0.7
RELATION_EXC_OBJ_NAMES = ['drawer handle', 'catapult button']
STATE_DICT = {
    'drawer': ['open', 'closed'],
    'cupboard': ['open', 'closed'],
    'mailbox': ['open', 'closed'],
    'catapult': ['triggered', 'not triggered']
}

# =========  Loaded Models ============
clip_classify=CLIP_classify()


def get_object_state(object_name, image):
    if object_name in STATE_DICT:
        states = STATE_DICT[object_name]

        state_descriptions = [f'the {object_name} is {state}' for state in states]

        return clip_classify.classify(state_descriptions, image).split()[-1]
    return None

def get_pcd_dist(pts_A, pts_B):
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(pts_A)
    pcd_B = o3d.geometry.PointCloud()
    pcd_B.points = o3d.utility.Vector3dVector(pts_B)

    dists = pcd_A.compute_point_cloud_distance(pcd_B)
    try:
        dist = np.min(np.array(dists))
    except:
        dist = np.inf
    return dist

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, scipy.spatial.Delaunay):
        hull = scipy.spatial.Delaunay(hull)

    return hull.find_simplex(p)>=0


def is_inside(src_pts, target_pts, thresh=0.5):
    try:
        hull = scipy.spatial.ConvexHull(target_pts)
    except:
        return False
    # print("vertices of hull: ", np.array(hull.vertices).shape)
    hull_vertices = np.array([[0,0,0]])
    for v in hull.vertices:
        hull_vertices = np.vstack((hull_vertices, np.array([target_pts[v,0], target_pts[v,1], target_pts[v,2]])))
    hull_vertices = hull_vertices[1:]

    num_src_pts = len(src_pts)
    # Don't want threshold to be too large (specially with more objects, like 4, 0.9*thresh becomes too large)
    thresh_obj_particles = thresh * num_src_pts
    src_points_in_hull = in_hull(src_pts, hull_vertices)
    # print("src pts in target, thresh: ", src_points_in_hull.sum(), thresh_obj_particles)
    if src_points_in_hull.sum() > thresh_obj_particles:
        return True
    else:
        return False

# def is_in_top_of(src_pos, target_box, thresh=0.9):
#     upper_plane_z = np.max(target_box[:, 2]) 
#     above_count = np.sum(src_pos[:, 2] > upper_condition_z) 
#     return (above_count / len(src_pos)) >= thresh
def is_on_top_of(src_pos, target_box):
    upper_plane_z = np.max(target_box[:, 2])
    return src_pos[2] > upper_plane_z  


class Node(object):
    def __init__(self, name, pcd):
        self.name = name
        self.pcd = pcd
        self.index = None
        boxes3d_pts = o3d.utility.Vector3dVector(self.pcd)
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(boxes3d_pts)
        self.bbox = bbox # 3d bounding box
        self.pos = bbox.get_center()
        self.corner_pts = np.array(bbox.get_box_points())
        self.state = None

    def set_index(self, index):
        self.index = index

    def set_state(self, state):
        self.state = state

    def __str__(self):
        return self.get_full_name_w_state()

    def __hash__(self):
        return hash(self.name, self.index)

    def __eq__(self, other):
        return True if (self.name == other.name and self.index == other.index) else False

    def get_full_name(self):
        name_w_index = f"{self.name} {self.index}" if self.index is not None and self.index!=1 else self.name
        return name_w_index

    def get_full_name_w_state(self):
        name_w_index = f"{self.name} {self.index}" if self.index is not None and self.index!=1 else self.name
        if self.state is not None:
            return f"{name_w_index} ({self.state})"
        else:
            return name_w_index


class Edge(object):
    def __init__(self, start_node, end_node, edge_type="None"):
        self.start = start_node
        self.end = end_node
        self.edge_type = edge_type
    
    def __hash__(self):
        return hash((self.start, self.end, self.edge_type))

    def __eq__(self, other):
        if self.start == other.start and self.end == other.end and self.edge_type == other.edge_type:
            return True
        else:
            return False

    def __str__(self):
        return self.start.get_full_name() + " -> " + self.edge_type + " -> " + self.end.get_full_name()


class SceneGraph(object):
    """
    Create a spatial scene graph
    """
    def __init__(self):
        self.nodes = []
        self.edges = {}
        self.node_indices = {}

    def add_node(self, new_node, image):
        self.add_node_wo_state(new_node)
        self.add_object_state(new_node, image)
    
    def add_node_wo_state(self, new_node):
        if new_node.name in self.node_indices:
            self.node_indices[new_node.name] += 1
        else:
            self.node_indices[new_node.name] = 1
        new_node.set_index(self.node_indices[new_node.name])

        for node in self.nodes:
            self._add_edge(node, new_node)
        self.nodes.append(new_node)

    def _add_edge(self, node, new_node):
        if new_node.name in RELATION_EXC_OBJ_NAMES or node.name in RELATION_EXC_OBJ_NAMES:
            return
        dist = get_pcd_dist(node.pcd, new_node.pcd)
        
        box_A_pts, box_B_pts = np.array(node.pcd), np.array(new_node.pcd)
        box_A, box_B = node.corner_pts, new_node.corner_pts
        pos_A, pos_B = node.pos, new_node.pos

        # IN CONTACT
        if dist < IN_CONTACT_DISTANCE:
            if is_inside(src_pts=box_B_pts, target_pts=box_A_pts, thresh=INSIDE_THRESH):
                self.edges[(new_node.get_full_name(), node.get_full_name())] = Edge(new_node, node, "inside")
            elif is_inside(src_pts=box_A_pts, target_pts=box_B_pts, thresh=INSIDE_THRESH):
                self.edges[(node.get_full_name(), new_node.get_full_name())] = Edge(node, new_node, "inside")
            elif is_on_top_of(src_pos=pos_B, target_box=box_A):
                self.edges[(new_node.get_full_name(), node.get_full_name())] = Edge(new_node, node, "on top of")
            elif is_on_top_of(src_pos=pos_A, target_box=box_B):
                self.edges[(node.get_full_name(), new_node.get_full_name())] = Edge(node, new_node, "on top of")
    
    def add_object_state(self, node, image):
        state = get_object_state(node.name, image)
        if state is not None:
            node.set_state(state)
        return node

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        res = "  [Nodes]:\n"
        for node in self.nodes:
            res += "    "
            res += (
            f'{node} -- '
            f'position: [{float(node.pos[0]):.2f}, {float(node.pos[1]):.2f}, {float(node.pos[2]):.2f}], '
            f'x_range: [{float(node.bbox.min_bound[0]):.2f}, {float(node.bbox.max_bound[0]):.2f}], '
            f'y_range: [{float(node.bbox.min_bound[1]):.2f}, {float(node.bbox.max_bound[1]):.2f}], '
            f'z_range: [{float(node.bbox.min_bound[2]):.2f}, {float(node.bbox.max_bound[2]):.2f}]'
            )
            res += "\n"
        res += "  [Edges]:\n"
        for edge_key, edge in self.edges.items():
            res += "    "
            res += str(edge)
            res += "\n"
        return res