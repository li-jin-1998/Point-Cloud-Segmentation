import numpy as np
import open3d as o3d
import plyfile

path = './test2.ply'
# v = read_obj(path)
plydata = plyfile.PlyData.read(path)
print(plydata)
# Get the number of points in the point cloud
num_points = plydata['vertex'].count
print("Number of points:", num_points)

# Get the point coordinates
points = plydata['vertex'].data['x'], plydata['vertex'].data['y'], plydata['vertex'].data['z']
colors = plydata['vertex'].data['red'], plydata['vertex'].data['green'], plydata['vertex'].data['blue']
# Convert the coordinates to a NumPy array
points = np.array([points[0], points[1], points[2]]).T
colors = np.array(colors).T
pc = np.hstack((points, colors))
print(pc.shape)
# points = np.random.rand(1000, 3)
print(points.shape)
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(pc)
#
# o3d.visualization.draw_geometries([point_cloud])
