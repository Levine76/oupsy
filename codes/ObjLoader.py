# -*-coding:Latin-1 -*
"""
Function to load object from file, ply or obj type
Obj file must be in ascii, binary not supported
"""

import numpy as np
import os
from codes.PointCloud import PointCloud
from plyfile import PlyData, PlyElement
import pickle

def load_point_cloud(path):
    filename, file_extension = os.path.splitext(path)
    if file_extension == ".ply":
        return load_ply(path)
    elif file_extension == ".obj":
        return load_obj(path)
    else:
        return None


def load_ply(path):
    try:
        plydata = PlyData.read(path)
        vertices = np.zeros((len(plydata['vertex'].data), 3), dtype=float)
        vertices[:,0] = plydata['vertex'].data[plydata['vertex'].data.dtype.names[0]]
        vertices[:,1] = plydata['vertex'].data[plydata['vertex'].data.dtype.names[1]]
        vertices[:,2] = plydata['vertex'].data[plydata['vertex'].data.dtype.names[2]]
        if len(plydata['vertex'].data[0]) == 6:
            colors = np.zeros((len(plydata['vertex'].data), 3), dtype=int)
            colors[:,0] = plydata['vertex'].data[plydata['vertex'].data.dtype.names[3]]
            colors[:,1] = plydata['vertex'].data[plydata['vertex'].data.dtype.names[4]]
            colors[:,2] = plydata['vertex'].data[plydata['vertex'].data.dtype.names[5]]
            return PointCloud(vertices, colors)
        else:
            return PointCloud(vertices)
    except:
        return None


def load_obj(path):
    vertices = []
    try:
        for line in open(path, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                vertices.append(tuple(map(float, values[1:4])))
        if len(vertices) == 0:
            return None
        return PointCloud(np.array(vertices))
    except UnicodeDecodeError :
        return pickle.load(open(path, "rb"))
    except:
        return None
