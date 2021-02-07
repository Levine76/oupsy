# -*-coding:Latin-1 -*
"""
Point cloud object information
Keeps points and colors.
"""
import numpy as np


class PointCloud:
    def __init__(self, vertices, colors=None):
        if colors is not None:
            if type(colors).__module__ != np.__name__:
                self.colors = np.array(colors)
            else:
                self.colors = colors
        else:
            self.colors = None

        if type(vertices).__module__ != np.__name__:
            self.vertices = np.array(vertices)
        else:
            self.vertices = vertices

    def copy(self):
        return PointCloud(np.copy(self.vertices), self.colors)

    def __str__(self):
        return "PointCloud : " + str(len(self.vertices)) + " vertices " + ("with" if self.colors else "without") + " colors"
