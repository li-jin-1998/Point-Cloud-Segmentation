import numpy as np


def grid_subsampling(data, grid_size):
    """
    Perform grid subsampling on input data.

    Parameters:
    - data: Input data, a numpy array with shape (n_samples, n_features).
    - grid_size: Grid size for subsampling, a tuple specifying the number of cells along each dimension.

    Returns:
    - subsampled_data: Subsampled data, a numpy array with shape (n_cells, n_features), where n_cells is the total number of cells after subsampling.
    """

    # Calculate number of cells along each dimension
    n_cells = [int(np.ceil(data.shape[i] / grid_size[i])) for i in range(len(grid_size))]

    # Initialize subsampled data array
    subsampled_data = np.zeros((np.prod(n_cells), data.shape[1]))

    # Iterate through each cell
    cell_index = 0
    for i in range(n_cells[0]):
        for j in range(n_cells[1]):
            # Define cell boundaries
            cell_start = [i * grid_size[0], j * grid_size[1]]
            cell_end = [(i + 1) * grid_size[0], (j + 1) * grid_size[1]]

            # Select samples within cell boundaries
            cell_samples = data[(data[:, 0] >= cell_start[0]) & (data[:, 0] < cell_end[0]) &
                                (data[:, 1] >= cell_start[1]) & (data[:, 1] < cell_end[1])]

            # If cell is not empty, compute mean of samples and assign to subsampled data
            if len(cell_samples) > 0:
                subsampled_data[cell_index] = np.mean(cell_samples, axis=0)

            cell_index += 1

    return subsampled_data


if __name__ == "__main__":
    from utils.functions import read_ply

    point_cloud, _, _ = read_ply('test2.ply')

    sampled_points = grid_subsampling(point_cloud, grid_size=(3, 3))

    # 打印采样后的点的数量
    print("采样后的点的数量:", sampled_points.shape[0])

    from farthest_point_sampling import visualize_points

    # 绘制原始点云
    visualize_points(point_cloud)

    # 绘制采样后的点云
    visualize_points(sampled_points)
