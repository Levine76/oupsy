# -*-coding:Latin-1 -*
import open3d
import numpy as np
import math
from codes import PointCloud


"""
Keep the point cloud data and colors.
Each point is associated with his array index.
"""


class TrainingSequence:
    def __init__(self):
        self.inputs = None
        self.dimension = 0
        self.center = None
        self.emax = 0.0
        self.inputs_size = 0
        self.normals = None

    """
    Init the point cloud representation
    """
    def init(self, inputs):
        if type(inputs).__module__ == PointCloud.__name__:
            self.inputs = inputs.vertices
            self.colors = inputs.colors
        elif type(inputs).__module__ != np.__name__:
            # Convert into matrix
            self.inputs = np.zeros((len(self.inputs), len(self.inputs[0])))
            self.colors = None
        else:
            self.inputs = inputs
            self.colors = None

        self.dimension = self.inputs.shape[1]
        self.center = None
        self.emax = 0.0
        self.inputs_size = len(self.inputs)

    def has_color(self):
        return self.colors is not None

    def mean_color(self, indices):
        if self.colors is None:
            return None
        colors = np.array(self.colors[indices], dtype=int)
        return np.array(colors.mean(0), dtype=int)

    def get_dimension(self):
        return self.dimension

    def get_center(self):
        return self.center

    def get_emax(self):
        return self.emax

    def get_all(self):
        return self.inputs

    """
    Compute the centroid of point cloud
    """
    def compute_center(self):
        # Compute the center of inputs 
        return self.inputs.mean(0)

    """
    Compute the centroid inputs of indices point cloud
    """
    def centroid_inputs(self, indices):
        inputs = np.array(self.inputs[indices])
        return inputs.mean(0)

    """
    Get point cloud size
    """
    def size(self):
        return self.inputs_size

    """
    Get point of point of the specified index
    """
    def get(self, index):
        return self.inputs[index]

    """
    Get list of points of specified indices
    """
    def retrieve(self, indices):
        return self.inputs[indices]


    def set(self, indices, values):
        self.inputs[indices] = values

    def compute_emax(self):
        emax = 0
        for id in range(self.inputs.shape[0]):
            squared_dist = np.vdot(self.inputs[id], self.inputs[id])
            if squared_dist > emax:
                emax = squared_dist
        return emax

    def compute_fake_emax(self):
        fake = np.max(np.abs(self.inputs)) + 0.01   # VR : Rq. calcul rayon de la boule contenant tous les points (+0.01 pour être sûr d'avoir un rayon suffisant)
        return fake * fake                          # VR : Rq. emax par dimension

    def compute_distortion(self):
        return np.vdot(self.inputs, self.inputs)

    def center_and_normalize(self):
        # Recenter
        self.center = self.compute_center()     # VR : Rq. centre = moyenne suivant chaque dimension
        self.inputs -= self.center

        # Normalize
        self.emax = self.compute_fake_emax()    # VR : Rq. emax (via métrique L2) par dimension
        self.inputs /= math.sqrt(self.emax)*2   # VR : Rq. F = b * rho * sqrt( emax ) avec rho=0.5 => ICI calcul de x*F/b (on ne tient pas encore compte de b)

    def translate(self, indices, translate):
        self.inputs[indices] += translate

    def scale(self, indices, factor):
        self.inputs[indices] *= factor

    def compute_normals(self):
        size = np.size(self.inputs, 0)  # Z: Or size = inputs.shape[0], the number of vertex
        pcd = open3d.geometry.PointCloud()   # Z: Creating an Open3D PointCloud object
        pcd.points = open3d.utility.Vector3dVector(self.inputs) # Z: Storing vertex imfprmation into this PointCloud object
        open3d.geometry.PointCloud.estimate_normals(pcd, open3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))    # Z: Estimating the normals. Searching neighbors within 0.01 from the query point and maximun 30 neighbors to estimate the normals.
        self.normals = np.asarray(pcd.normals)[:size, :]    # Z: Extracting the normals of all vertices and storing them into normals.
        return self.normals
