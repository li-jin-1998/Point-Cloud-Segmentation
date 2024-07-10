import numpy as np
from sklearn.neighbors import BallTree


def ball_nearest_neighbors(pts_src, pts_dest, data_src):
    """
    寻找最近邻对应。给定源点集和目标点集，以及源点的数据集，此函数通过BallTree算法寻找每个目标点最近的源点，并返回其对应的数据。

    :param pts_src: 源点集，numpy数组，形状为(N, D)，N为点的数量，D为维度。
    :param pts_dest: 目标点集，numpy数组，形状为(M, D)，M为点的数量，D为维度。
    :param data_src: 源点的数据集，numpy数组，形状为(N, K)，N为点的数量，K为每个点的数据维度。
    :return: 目标点对应的最近源点的数据，numpy数组，形状为(M, K)。
    """
    try:
        # 构建BallTree
        tree = BallTree(pts_src, leaf_size=5)

        # 执行查询
        _, indices = tree.query(pts_dest, k=1)
        indices = indices.ravel()

        # 索引数据
        data_dest = data_src[indices]

    except Exception as e:
        # 异常处理，可以选择记录日志或抛出更具体的异常
        raise RuntimeError("在执行近邻搜索过程中遇到错误") from e

    return data_dest


def ballquery(points, query_point, radius):
    """
    Ball query operation to find neighbors within a given radius.

    Parameters:
        points (numpy array): Point cloud data of shape (N, D), where N is the number of points and D is the dimensionality.
        query_point (numpy array): Query point coordinates.
        radius (float): Radius within which to find neighbors.

    Returns:
        neighbors_indices (list): List of indices of neighbor points within the radius.
    """
    neighbors_indices = []
    for i, point in enumerate(points):
        distance = np.linalg.norm(point - query_point)
        if distance <= radius:
            neighbors_indices.append(i)
    return neighbors_indices


if __name__ == '__main__':
    # Example usage:
    # Assume points is a numpy array representing the point cloud data of shape (N, D)
    points = np.array([[0, 0],
                       [1, 0],
                       [0, 1],
                       [1, 1]])
    query_point = np.array([0.5, 0.5])
    radius = 0.8
    neighbors_indices = ballquery(points, query_point, radius)
    print("Indices of neighbors within radius:", neighbors_indices)
