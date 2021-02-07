# -*-coding:Latin-1 -*
"""
Store information for each node in the tree and temporary computation results
"""

from codes.TrainingSequence import TrainingSequence
import math
import numpy as np
from codes.VoronoiCell import VoronoiCell
from codes.VoronoiCellNode import VoronoiCellNode
from codes import shortest

class CompleteNode:
    def __init__(self, representant, inputs, factor=1, parent=None, depth=0, isLeaf=True):
        self.parent = parent
        self.depth = depth
        self.isLeaf = isLeaf
        self.representant = representant
        self.factor = factor
        self.methodID = -1      # VR : the id of quantization method
        self.b = -1             # VR : store the b for each node
        self.id = -1
        self.partition_state = []
        self.geometric_center = None

        self.inputs = inputs
        self.children = []

        self.lambdaScore = 0.0
        self.deltaRate = 0.0
        self.deltaDistortion = 0.0
        self.probabilityOccurency = 0
        self.distortion = 0.0
        self.rate = 0.0

    def see_inputs (self):
        print("node with id = ",self.id," has inputs = ", self.inputs)


    def quantization_clone(self):
        node = CompleteNode(self.representant,
            self.inputs,
            self.factor,
            self.parent,
            self.depth,
            self.isLeaf)
        node.distortion = self.distortion
        node.rate = self.rate
        node.probabilityOccurency = self.probabilityOccurency
        return node

    def save_quantization(self, quantized_node):
        self.children = quantized_node.children
        self.lambdaScore = quantized_node.lambdaScore
        self.deltaRate = quantized_node.deltaRate
        self.deltaDistortion = quantized_node.deltaDistortion
        self.probabilityOccurency = quantized_node.probabilityOccurency
        self.distortion = quantized_node.distortion
        self.rate = quantized_node.rate
        self.methodID = quantized_node.methodID

        for child in self.children:
            child.parent = self

    """Computation"""
    def compute_distortion(self, ts):
        ## The following is the Distortion Method for start_Z_planes
        inputs = ts.retrieve(self.inputs)
        # print("...",inputs)
        if inputs.shape[0] >= 3:
            C = shortest.best_plane(inputs, 1)
            sumdis = 0
            for i in range(inputs.shape[0]):
                dis = shortest.shortest_distance(inputs[i][0], inputs[i][1], inputs[i][2], C[0], C[1], C[2])
                sumdis += dis
            sumdis = sumdis**2
            # mean_sumdis = sumdis / inputs.shape[0]
            # mean_sumdis = mean_sumdis**2
            center = [0, 0, 0]
            mean_center = inputs.mean(0)
            dev = mean_center - center
            return (sumdis + np.vdot(dev, dev)*inputs.shape[0]) / (self.factor * self.factor)
            # return (mean_sumdis + np.vdot(dev, dev)) / (self.factor * self.factor)
        else:
            center = inputs.mean(0)
            res = inputs - center
            return np.vdot(res, res) / (self.factor * self.factor)

    def compute_rate(self):
        if self.probabilityOccurency == 0:
            return 0.0
        else:
            return -math.log(self.probabilityOccurency, 2)

    def compute_delta_distortion(self):
        child_distortion = 0
        for child in self.children:
            child_distortion += child.distortion
        return self.distortion - child_distortion

    # this function is used only to compute delta rate of pending leaves.
    def compute_delta_rate(self,partition_state,pro):

            pro.add_state(partition_state)

            # obtain the probability to calculate the rate of leaf clone
            self.probabilityOccurency = pro.get_probability (partition_state)

            # restore the probability array since current leaf is temporarily being checked and not yet decided to be the best leaf
            pro.sub_state(partition_state)

            return self.compute_rate()

    def compute_lambda(self):
        return self.deltaDistortion / self.deltaRate

# used only for the children of the leaf_clone being quantized in quantize_pending_leaves
    def update_distortion_rate(self, ts):
        self.probabilityOccurency = 0
        self.distortion = self.compute_distortion(ts)
        self.rate = 0

# used only for updating the lambda of leaf_clone being quantized in quantize_pending_leaves
    def update_lambda(self, ts, partition_state, pro):
        for child in self.children:
            child.update_distortion_rate(ts)
        self.deltaDistortion = self.compute_delta_distortion()
        self.deltaRate = self.compute_delta_rate(partition_state,pro)
        if self.deltaRate != 0:
            self.lambdaScore = self.compute_lambda()
        else:
            self.lambdaScore = self.deltaDistortion

# used for all the leaves present in lambda sorted since probability of all partitions has changed due to total partitions change
    def update_lambda_score (self,pro):
        partition_state = self.parent.partition_state
        self.probabilityOccurency = pro.get_probability(partition_state)
        self.deltaRate = self.compute_rate()
        if self.deltaRate == 0:
            return
        else:
            self.lambdaScore = self.compute_lambda()

    def compute_emax(self, ts):
        emax = 0
        inputs = ts.retrieve(self.inputs)
        for id in range(inputs.shape[0]):
            squared_dist = np.vdot(inputs[id], inputs[id])
            if squared_dist > emax:
                emax = squared_dist
        return emax

    """Create representation"""

    """
    Use to create the list of Voronoi cell
    """
    def get_voronoiCell(self, quantization, center, max_depth=-1):
        dim = np.array([1/self.factor] * len(center))
        voronoiCells = [VoronoiCell(center, len(center))]

        if self.depth != max_depth:
            if not self.isLeaf:
                for child in self.children:
                    lattice = quantization.methods[self.methodID].from_index_to_pos_parent(len(center), self.representant)
                    child_center = center + lattice / self.factor
                    voronoiCells.extend(child.get_voronoiCell(dim, quantization, child_center, max_depth))
        return voronoiCells

    """
    Use to create Voronoi cell tree structure
    """
    def get_voronoiCellNode(self, quantization, center, max_depth=-1):
        dim = np.array([1/self.factor] * len(center))
        children = []
        # print("voronoi cell center is = ",center)

        if self.depth != max_depth:
            if not self.isLeaf:
                for child in self.children:
                    lattice = quantization.methods[self.methodID].from_index_to_pos_parent(len(center), child.representant)
                    # print("lattice is = ", lattice)
                    child_center = center + lattice / self.factor
                    # print("child center is = ",child_center)
                    children.append(child.get_voronoiCellNode(quantization, child_center, max_depth))
        return VoronoiCellNode(center, dim, children, self)