import pyvista as pv
from sklearn.neighbors import KDTree

file_path = './test2.ply'
mesh = pv.read(file_path)
points = mesh.points
colors = mesh.active_scalars

# Build KD-tree
kdtree = KDTree(points, leaf_size=30)

# Define the query point and radius
query_point = points[0]
radius = 0.2

# Query neighbors within the radius
indices = kdtree.query_radius([query_point], r=radius)

# Print the indices of neighbors within the radius
print("Indices of neighbors within radius:", indices[0])

# Access neighbor points using the indices
neighbor_points = points[indices[0]]

# Print the neighbor points
print("Neighbor points:", neighbor_points)
