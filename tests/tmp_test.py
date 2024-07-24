import numpy as np
from scipy.spatial import Delaunay

def alpha_shape_3d(points, alpha):
    """
    Compute the alpha shape (concave hull) for 3D points.
    @param points: np.array of shape (n,3) points.
    @param alpha: alpha value.
    @return: indices of points forming the alpha shape.
    """
    if len(points) < 5:
        # If there are 4 points or less, the convex hull is the best we can do
        return np.array(range(len(points)))

    tri = Delaunay(points)
    edges = set()
    edge_points = []

    # Loop over tetrahedra
    # indices of vertices of each tetrahedron
    for tetra in tri.simplices:
        # Extract vertices of each tetrahedron
        pa, pb, pc, pd = points[tetra]
        
        # Calculate the circumradius and the volume of the tetrahedron
        vol, circum_radius = tetrahedron_volume_circumradius(pa, pb, pc, pd)
        if vol == 0:
            continue  # Degenerate tetrahedron, skip

        if circum_radius < alpha:
            # Add all edges of this tetrahedron to the set
            for edge in itertools.combinations(tetra, 2):
                edges.add(tuple(sorted(edge)))

    for edge in edges:
        edge_points.extend(edge)

    return np.unique(edge_points)

def tetrahedron_volume_circumradius(a, b, c, d):
    """
    Calculate the volume and circumradius of the tetrahedron defined by points a, b, c, d.
    @return: volume and circumradius
    """
    A = np.array([a, b, c, d])
    V = np.vstack([A - d, np.ones((1, 4))])
    volume = np.abs(np.linalg.det(V)) / 6

    if volume == 0:
        return 0, float('inf')  # Degenerate tetrahedron

    # Calculate the circumradius
    a = np.linalg.norm(b - c)
    b = np.linalg.norm(a - c)
    c = np.linalg.norm(a - b)
    circum_radius = a * b * c / (4 * volume)

    return volume, circum_radius

# Example usage
points = np.random.rand(100, 3) * 50  # Generate some random points in 3D
alpha = 10.0  # Set a suitable alpha value
indices = alpha_shape_3d(points, alpha)
surface_points = points[indices]

print("Points on the alpha shape:", surface_points)
