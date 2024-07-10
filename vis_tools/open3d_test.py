import numpy as np
import open3d as o3d
import plyfile

path = '../test2.ply'
plydata = plyfile.PlyData.read(path)
print(plydata)
# Get the number of points in the point cloud
num_points = plydata['vertex'].count
print("Number of points:", num_points)

# Get the point coordinates
points = plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']
colors = plydata['vertex'].data['red'], plydata['vertex'].data['green'], plydata['vertex'].data['blue']
# Convert the coordinates to a NumPy array
points = np.array(points).T
colors = np.array(colors).T
pc = np.hstack((points, colors))
print(pc.shape)
print(points.shape, colors.shape)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
# point_cloud.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([point_cloud])
