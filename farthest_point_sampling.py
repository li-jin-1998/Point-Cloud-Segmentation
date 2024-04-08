import numpy as np
import pyvista as pv


def distance(point, points):
    distances = np.sum((points - point) ** 2, axis=1)
    min_index = np.argmin(distances)
    min_distance = np.sqrt(distances[min_index])
    return min_index, min_distance


def farthest_point_sampling(point_cloud, num_points):
    num_points_cloud = point_cloud.shape[0]
    sampled_indices = []  # 保存采样点的索引
    sampled_points = []  # 保存采样点的坐标

    # 随机选择一个起始点
    start_index = np.random.randint(num_points_cloud)
    sampled_indices.append(start_index)
    sampled_points.append(point_cloud[start_index])

    # 逐步添加剩余的点
    for _ in range(num_points - 1):
        farthest_distance = -1
        farthest_index = -1

        # 找到距离已选点最远的点
        for i, point in enumerate(point_cloud):
            if i not in sampled_indices:
                r, min_distance = distance(point, np.array(sampled_points))
                if min_distance > farthest_distance:
                    farthest_distance = min_distance
                    farthest_index = i

        # 将最远点添加到已选点中
        sampled_indices.append(farthest_index)
        sampled_points.append(point_cloud[farthest_index])

    return np.array(sampled_points)


def visualize_points(points):
    p = pv.Plotter()
    p.add_points(points, opacity=1.0, point_size=3, render_points_as_spheres=True)
    p.show(cpos='xy')


if __name__ == "__main__":
    from utils.functions import read_ply

    point_cloud, _, _ = read_ply('test2.ply')

    # 采样后的点的数量
    num_points_sampled = 100

    # 进行最远点采样
    sampled_points = farthest_point_sampling(point_cloud, num_points_sampled)

    # 打印采样后的点的数量
    print("采样后的点的数量:", sampled_points.shape[0])

    # 绘制原始点云
    visualize_points(point_cloud)

    # 绘制采样后的点云
    visualize_points(sampled_points)
