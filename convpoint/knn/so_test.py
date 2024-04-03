import ctypes

# so 文件名称
# grid_subsampling.cpython-39-x86_64-linux-gnu
# nearest_neighbors.cpython-39-x86_64-linux-gnu
lib = ctypes.CDLL("./cpp/nearest_neighbors.cpython-39-x86_64-linux-gnu.so")

# 找到所需的方法
# multiply = lib.multiply

# 设定方法的参数和返回类型
# multiply.argtypes = (ctypes.c_int, ctypes.c_int)
# multiply.restype = ctypes.c_int
#
# # 调用方法
# result = multiply(2, 3)

# print(result)