import numpy as np


def poisson_disk_sampling_2d(points, min_distance, num_samples=30):
    """
    二维泊松盘采样

    参数：
    - points: 形状为 (N, 2) 的 numpy 数组，表示 N 个点的 2 维坐标
    - min_distance: 两个点之间的最小距离
    - num_samples: 每次迭代尝试生成的点的数量

    返回值：
    - samples: 形状为 (N, 2) 的 numpy 数组，表示生成的点集
    """
    cell_size = min_distance / np.sqrt(2)  # 确保每个点周围有足够的空间
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid = np.full((grid_width, grid_height), -1)
    samples = []
    active = []

    def in_bounds(point):
        return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y

    def get_neighbors(point):
        x, y = point
        candidates = [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
        return [(nx, ny) for nx, ny in candidates if in_bounds((nx, ny))]

    def get_random_point_around(point):
        for _ in range(num_samples):
            angle = np.random.rand() * 2 * np.pi
            radius = np.random.rand() * min_distance + min_distance
            new_point = (point[0] + radius * np.cos(angle), point[1] + radius * np.sin(angle))
            if in_bounds(new_point):
                return new_point
        return None

    def is_valid(point):
        if not in_bounds(point):
            return False
        px = int((point[0] - min_x) / cell_size)
        py = int((point[1] - min_y) / cell_size)
        min_px = max(0, px - 1)
        max_px = min(grid_width - 1, px + 1)
        min_py = max(0, py - 1)
        max_py = min(grid_height - 1, py + 1)
        for i in range(min_px, max_px + 1):
            for j in range(min_py, max_py + 1):
                if grid[i, j] != -1:
                    if np.linalg.norm(np.array(samples[grid[i, j]]) - np.array(point)) < min_distance:
                        return False
        return True

    for point in points:
        if in_bounds(point):
            px = int((point[0] - min_x) / cell_size)
            py = int((point[1] - min_y) / cell_size)
            grid[px, py] = len(samples)
            samples.append(point)
            active.append(point)

    while active:
        index = np.random.randint(len(active))
        current_point = active[index]
        found_valid_point = False
        for _ in range(num_samples):
            new_point = get_random_point_around(current_point)
            if new_point and is_valid(new_point):
                px = int((new_point[0] - min_x) / cell_size)
                py = int((new_point[1] - min_y) / cell_size)
                grid[px, py] = len(samples)
                samples.append(new_point)
                active.append(new_point)
                found_valid_point = True
                break
        if not found_valid_point:
            active.pop(index)

    return np.array(samples)


def project_to_2d(point_cloud):
    """
    将三维点云投影到二维平面

    参数：
    - point_cloud: 形状为 (N, 3) 的 numpy 数组，表示 N 个点的 3 维坐标

    返回值：
    - projected_points: 形状为 (N, 2) 的 numpy 数组，表示投影后的点云
    """
    return point_cloud[:, :2]


def project_to_3d(projected_points, z_values):
    """
    将二维点云和z值映射回三维空间

    参数：
    - projected_points: 形状为 (N, 2) 的 numpy 数组，表示投影后的点云
    - z_values: 形状为 (N,) 的 numpy 数组，表示每个点的z坐标

    返回值：
    - point_cloud_3d: 形状为 (N, 3) 的 numpy 数组，表示映射回的三维点云
    """
    return np.column_stack((projected_points[:1000], z_values[:1000]))


import pyvista as pv


def visualize_points(points):
    p = pv.Plotter()
    p.add_points(points, opacity=1.0, point_size=3, render_points_as_spheres=True)
    p.show(cpos='xy')


# 示例用法
if __name__ == "__main__":
    from utils.functions import read_ply

    point_cloud_3d, _, _ = read_ply('../test.ply')
    # 投影到二维平面
    projected_points = project_to_2d(point_cloud_3d)
    print(projected_points.shape[0])
    # 进行二维泊松盘采样
    min_distance = 4
    sampled_points_2d = poisson_disk_sampling_2d(projected_points, min_distance)
    print(sampled_points_2d.shape[0])
    # 映射回三维空间
    sampled_points_3d = project_to_3d(sampled_points_2d, point_cloud_3d[:, 2])

    # 打印采样后的点的数量
    print("采样后的点的数量:", sampled_points_3d.shape[0])

    # 绘制原始点云
    visualize_points(point_cloud_3d)

    # 绘制采样后的点云
    visualize_points(sampled_points_3d)
