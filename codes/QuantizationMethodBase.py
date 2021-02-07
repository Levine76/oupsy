"""
Quantization method used put in comparison with other quantization method
b correspond to the lattice dimension, b=2 (2x2x2) and b=3 (3x3x3)
"""
from codes.CompleteNode import CompleteNode
from codes.TrainingSequence import TrainingSequence
from numpy import argwhere
import numpy as np
import math

class QuantizationMethodBase:
    def __init__(self, b):
        self.b = b                          # VR : le facteur de projection au sein du lattice tronque
        self.add = ((self.b + 1)%2)*0.5     # VR : la translation au "centre" de ce lattice tronque

    def quantize_node(self, ts, node):
        assert(len(node.inputs) != 0)

        # Normalize inputs by factor
        self.scale(ts, node.inputs)

        # Center to the center of the lattice representation
        self.center(ts, node.inputs)

        # Retrieve indices for inputs
        # quantized result
        # this result has a list of ids of representants corresponding to each input point
        result = self.quantize_base(ts.retrieve(node.inputs))

        # print ("result of quantize base = ",result)

        # Store the resulting nodes
        max_id = int(math.pow(self.b, ts.get_dimension()))
        # print("max id is:- ",max_id)
        for id in range(max_id):
            # Retrieve the corresponding indices from the inputs array
            indices_from_inputs = np.ndarray.flatten(np.argwhere(result == id))

            # print ("indices from inputs are :- ",indices_from_inputs)

            # Check if there are inputs inside this leaf
            if indices_from_inputs.shape[0] != 0:
                # Get inputs
                child_inputs = node.inputs[indices_from_inputs]
                # print("region with input points is:- ",id)

                # Translate data to center of node
                center = self.from_index_to_pos(ts.get_dimension(), id)
                translate = - center - 0.5 # Remove 0.5 to be in the lattice coordinate
                ts.translate(child_inputs, translate)

                # Create the leaf
                child = CompleteNode(id,
                                     child_inputs,
                                     node.factor * self.b,
                                     node,
                                     node.depth+1)
                node.children.append(child)
                # print ("representant of child is :- ", child.representant)


    def apply_quantization(self, ts, child):
        # Normalize
        self.scale(ts, child.inputs)

        # Center in child coordinate
        center = self.from_index_to_pos(ts.get_dimension(), child.representant)
        translate = - center + (self.b/2 - 0.5) # Center to center + Center in lattices coordinates
        ts.translate(child.inputs, translate)

    def center(self, ts, indices):
        # 0 should be at left point of the grid to make easier quantization
        ts.translate(indices, self.b/2.0)

    def scale(self, ts, indices):
        ts.scale(indices, self.b)

    def quantize_base(self, vec):
        # Find indices for vector indices
        result = np.floor(vec)
        # result actually has the list of input points after applying greatest integer function on each co-ordinate
        vec = np.power(self.b, np.arange(0, result.shape[1]))
        result *= vec
        # print("new result = ", result)
        return np.sum(result, axis=1)

    def from_pos_to_index(self, lattice_pos):
        # Get index from lattice vector in children coordinates
        return self.from_pos_to_indices(lattice_pos)

    def from_pos_to_indices(self, lattices_pos):
        # Get indices from lattices vector in children coordinates
        lattices_pos += 0.5
        return self.quantize_base(lattices_pos)

    def from_index_to_pos(self, dim, index):
        # Get lattice pos in children coordinates, from index
        return self.from_indices_to_pos(dim, np.array([index]))[0]

    def from_index_to_pos_parent(self, dim, index):
        # Get lattice pos in parent coordinates, from index
        return (self.from_indices_to_pos(dim, np.array([index]))[0] - self.b/2 + 0.5)/self.b

    def from_indices_to_pos(self, dim, indices):
        # Get lattices pos in children coordinates, from indices
        vec = np.power(self.b, np.arange(0, dim))
        indices_vec = np.copy(indices)
        result = np.zeros((len(indices_vec), dim))
        for d in range(dim - 1, -1, -1):
            # Find the value at this dimension
            result[:, d] = np.floor(indices_vec/vec[d])
            # Remove found value to the resulting dim
            indices_vec -= int(result[:, d] * vec[d])
        return result
