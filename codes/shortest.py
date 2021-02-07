# Python program to find the Perpendicular(shortest)
# distance between a point and a Plane in 3 D.

import math
import numpy as np
import scipy.linalg

def best_plane(data, order):
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]  #AM np.ones( number of rows)
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  #AM coefficients a, b ,c dans C
        # evaluate it on grid
        #Z = C[0] * X + C[1] * Y + C[2]  #AM the equation of the plane is z= ax + by + d
        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
        # evaluate it on a grid
        #Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)
    return C

def shortest_distance(x, y, z, a, b, c):
    d = abs((a * x + b * y - z + c))
    e = (math.sqrt(a * a + b * b + (-1) * (-1)))
    dis = d / e
    return dis

def write_ply(points_plot, path):
    head = "ply\n"+\
        "format ascii 1.0\n"+\
        "comment written by quantization_proc\n"+\
        "element vertex "+ str(len(points_plot)) + "\n"+\
        "property float32 x\n"+\
        "property float32 y\n"+\
        "property float32 z\n"+\
        "property uchar red\n"+\
        "property uchar green\n"+\
        "property uchar blue\n"+\
        "end_header\n"

    body = ""
    for id in range(len(points_plot)):
        point = str(points_plot[id,0]) + " " + str(points_plot[id,1]) + " " + str(points_plot[id,2]) + " " \
                + str(int(points_plot[id,3])) + " " + str(int(points_plot[id,4])) + " " + str(int(points_plot[id,5])) + "\n"
        body += point

    file = open(path, "w")
    file.write(head)
    file.write(body)
    file.close()