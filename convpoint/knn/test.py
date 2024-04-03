import numpy as np
import importlib.machinery
print(importlib.machinery.all_suffixes())
import cpp.nearest_neighbors as nearest_neighbors
# import cpp.radius_neighbors as cpp_neighbors
import time
import torch

from sklearn.neighbors import BallTree
#(2048, 3) (2753, 3) (2048, 4)

npts = 8192
K = 16
data = np.random.rand(npts,3).astype(np.float32)
data = torch.from_numpy(data).float().cuda()
data = data + 0

start = time.time()
tree = BallTree(data.cpu().numpy(), leaf_size=2)
_, indices = tree.query(data.cpu().numpy(), k=K)
print(time.time() - start)

start = time.time()
indices = nearest_neighbors.knn(data.cpu().numpy(), data.cpu().numpy(), K)
print(time.time() - start)

start = time.time()
indices = nearest_neighbors.knn(data.cpu().numpy(), data.cpu().numpy(), K, omp=True)
print(time.time() - start)


data = np.random.rand(5, npts, 3).astype(np.float32)
data = torch.from_numpy(data).float()

start = time.time()
indices = nearest_neighbors.knn_batch(data.cpu().numpy(), data.cpu().numpy(), K)
print(time.time() - start)

start = time.time()
indices = nearest_neighbors.knn_batch(data.cpu().numpy(), data.cpu().numpy(), K, omp=True)
print(time.time() - start)

start = time.time()
indices, queries = nearest_neighbors.knn_batch_distance_pick(data.cpu().numpy(), 256, K, omp=True)
print(time.time() - start)
print(queries.shape)
