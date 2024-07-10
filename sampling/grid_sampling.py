# Subsampling extension
import convpoint.knn.cpp.grid_subsampling as cpp_subsampling
import convpoint.knn.cpp.radius_neighbors as cpp_neighbors
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3,
                  t7 - t9,
                  t11 + t12,
                  t7 + t9,
                  t1 + t2 * t15,
                  t19 - t20,
                  t11 - t12,
                  t19 + t20,
                  t1 + t2 * t24], axis=1)

    return np.reshape(R, (-1, 3, 3))


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending on the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending on the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Sunsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                                batches_len,
                                                                                features=features,
                                                                                classes=labels,
                                                                                sampleDl=sampleDl,
                                                                                max_p=max_p,
                                                                                verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)


if __name__ == '__main__':
    # axis = np.array([[0, 1, 0]])
    # angle = np.array([np.pi / 2])
    # rotation_matrix = create_3D_rotations(axis, angle).astype(int)
    #
    # expected_matrix = np.array([[[0, 0, 1],
    #                              [0, 1, 0],
    #                              [-1, 0, 0]]])
    # print(rotation_matrix == expected_matrix)

    from utils.functions import read_ply

    file_path = '../test2.ply'
    ply_data = read_ply(file_path)
    points = ply_data[0]
    colors = ply_data[1]
    # new_column = np.ones((colors.shape[0], 1))
    # colors = np.hstack((colors, new_column))  # 使用 hstack 函数将新列添加到原始数组中
    # colors = np.asarray(colors, dtype=np.float32)
    labels = np.arange(ply_data[2]).reshape(-1, 1).astype(np.int32)
    print(points.shape, labels.shape)
    sampled_points, sampled_colors = grid_subsampling(points, labels=labels, sampleDl=0.2, verbose=0)
    print(sampled_colors)
    sampled_colors = colors[sampled_colors]
    sampled_colors = np.squeeze(sampled_colors, axis=1)
    # indices = batch_neighbors(points, points,[1024],[1024], radius=0.2)
    # print(indices)

    # choice = np.random.choice(len(mesh.points), 9000, replace=True)
    # print(np.unique(choice).shape)
    # sampled_points = points[choice]
    # sampled_colors = colors[choice]

    print(points.shape, sampled_points.shape, sampled_colors.shape)
    # pl = pv.Plotter(shape=(1, 2))
    # pl.subplot(0, 0)
    # pl.add_points(points, scalars=colors, point_size=5, style="points", rgb=True)
    # pl.add_title(str(points.shape[0]), font_size=20, font='times')
    # pl.view_xy()
    # pl.subplot(0, 1)
    # pl.add_points(sampled_points, scalars=sampled_colors, point_size=5, style="points", rgb=True)
    # pl.add_title(str(sampled_points.shape[0]), font_size=20, font='times')
    # pl.view_xy()
    # pl.show()

    from vis_tools.visualization import VisualizationPoints

    vis = VisualizationPoints(points_list=[points, sampled_points], colors_list=[colors, sampled_colors])
    vis.display()
