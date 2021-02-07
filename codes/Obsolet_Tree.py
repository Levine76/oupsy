from TrainingSequence import TrainingSequence
from statistics import mean
from CompleteNode import CompleteNode
import numpy as np
import operator
"""
Tree structure used for creating the point cloud quantization
"""
class Tree:
    def __init__(self, trainingSequence):
        self.distortion = 0.0
        self.start_distortion = 0.0
        self.rate = 0.0
        self.threshold_distortion = 0.0
        self.threshold_rate = 0.0
        self.nodes = []
        self.lambda_sorted = {}
        self.leaves = []
        self.counter = 0

        self.trainingSequence = trainingSequence
        # stream and mean points are in increasing depth wise order
        # stream contains only the list of nodes and not any leaf
        self.stream = []
        self.mean_points = []
        # Store leaves not already treated
        self.leaves_pending = []

        self.added_leaves = []
        self.current_leaf = 0

    def clear(self):
        self.distortion = 0.0
        self.rate = 0.0
        self.threshold_distortion = 0.0
        self.threshold_rate = 0.0
        self.nodes = []
        self.lambda_sorted = {}
        self.leaves = []

        # Store leaves not already treated
        self.leaves_pending = []

    def write(self, path):
        pass

    def load(self, path):
        pass

    def get_node(self, id):
        return self.nodes[id]

    def add_node(self, node):
        self.nodes.append(node)
        return len(self.nodes) - 1
    
    def pop_max_lambda(self):
        if len(self.lambda_sorted) == 0:
            return -1
        max_lambda = max(self.lambda_sorted)
        id = self.lambda_sorted[max_lambda].pop()
        if len(self.lambda_sorted[max_lambda]) == 0:
            self.lambda_sorted.pop(max_lambda, None)
        return id

    def pop_pending_leave(self):
        return self.leaves_pending.pop()

    def pending_leaves_empty(self):
        return len(self.leaves_pending) == 0

    def insert_lambda(self, lambdaScore, id):
        # Insert lambda
        if lambdaScore in self.lambda_sorted:
            self.lambda_sorted[lambdaScore].append(id)
        else:
            self.lambda_sorted[lambdaScore] = [id]

    def add_leaf(self, leaf):
        self.nodes.append(leaf)
        leaf.id = len(self.nodes) - 1
        id = leaf.id
        self.leaves.append(id)
        self.leaves_pending.append(id)

    def create_root(self):
        root = CompleteNode(0, np.arange(0, self.trainingSequence.size()))
        self.trainingSequence.center_and_normalize()
        self.add_leaf(root)
        root.update_distortion_rate(self.trainingSequence)
        self.distortion = root.distortion
        self.start_distortion = root.distortion
        self.rate = root.rate

    def normalized_distortion(self):
        return 100 * self.distortion / self.start_distortion

    def size(self):
        return len(self.leaves)

    def all_zeros(self,par):
        for i in par:
            if i != 0:
                return False
        return True

    def generate_stream (self,pro,qt):
        for i in self.nodes:
            node_id = i.id
            node_depth = i.depth
            node_state = i.partition_state
            if node_state == []:
                continue
            if node_id == 0:
                node_parent = 0
            else:
                node_parent = i.parent.id
            index = pro.get_index(node_state)
            # this is for nodes whose at least one child is splitted
            if not self.all_zeros(node_state):
                for j in pro.probability_table[index]:
                    if j[0] == node_state:
                        self.stream.append([node_id,node_depth,node_state,j[3],node_parent])
                        self.counter = self.counter + 1
            # this is for nodes whose all children are leaves
            if self.all_zeros(node_state):
                method = i.methodID
                quantization_method = qt.methods[method].b
                empty_partitions = quantization_method ** 3
                self.stream.append([node_id,node_depth,[],[-1,empty_partitions],node_parent])
                self.counter = self.counter + 1
        self.stream = sorted(self.stream, key=operator.itemgetter(1))
        return self.stream


    def get_mean_points (self):
        for i in self.nodes:
            if i.isLeaf:
                node_id = i.id
                node_depth = i.depth
                self.mean_points.append([node_id,node_depth,mean(i.inputs)])

        self.mean_points = sorted(self.mean_points, key = operator.itemgetter(1))
        return self.mean_points


