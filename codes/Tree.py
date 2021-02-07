from codes.TrainingSequence import TrainingSequence
from statistics import mean
from codes.CompleteNode import CompleteNode
import numpy as np
import operator
#import TreeDisplay
import copy
import math
import pickle
import os
import glob
from codes import ObjLoader
from codes.TrainingSequence import TrainingSequence
from codes import Quantization
from codes.QuantizationMethodBase import QuantizationMethodBase
import sys
from codes.Probability import Probability
import codes.ply_to_tsplvq
"""
Tree structure used for creating the point cloud quantization
"""
class Tree:
    def __init__(self, trainingSequence=None):
        self.distortion = 0.0
        self.start_distortion = 0.0
        self.rate = 0.0
        self.threshold_distortion = 0.0
        self.threshold_rate = 0.0
        self.nodes = []
        self.lambda_sorted = {}
        self.leaves = []
        self.counter = 0

        if trainingSequence:
            self.trainingSequence = trainingSequence
        else:
            self.trainingSequence = None
        # stream and mean points are in increasing depth wise order
        # stream contains only the list of nodes and not any leaf
        self.stream = []
        self.mean_points = []
        # Store leaves not already treated
        self.leaves_pending = []

        #Display tree
        #self.displayTree = TreeDisplay.TreeDisplay()

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
        os.makedirs(path[:-len(os.path.basename(path))], exist_ok=True)
        pickle.dump(self, open(path, "wb"))

    #Static method
    def load(path, methods=[2], rateMax=30000):
        if path[-4:].upper() == ".OBJ":
            tree = pickle.load(open(path, "rb"))
        elif path[-4:].upper() == ".PLY":
            tree = Tree.loadFromPly(path, methods=methods, rateMax=rateMax)
        else:
            raise Exception("File extension of " + path + " not supported to load tree")
        return tree

    def loadFromPly(file, methods=[2], rateMax=30000):
        return ply_to_tsplvq.ply_to_tsplvq(file, methods=methods, rateMax=rateMax, dest=None)

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
        id = self.add_node_id(leaf)
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

    def rec(self,nodeA,nodeB,nodeAdd,tree_compare, isRoot = False) :
            if((nodeA.id != 0 and nodeB.id != 0) or isRoot) :
                b_a = nodeA.b
                b_b = nodeB.b
                tree_compare.add_node_id(nodeAdd)
                
                tree_compare.displayTree.add_graph_node(nodeAdd)
                if(b_a != -1 and b_b != -1) :
                    if(b_a == b_b) :#On regarde si ils ont le mÃªme dÃ©coupage
                        children_a_id = [0] * (b_a ** 3)
                        for children in nodeA.children :
                            children_a_id[children.representant] = children.id

                        children_b_id = [0] * (b_b ** 3)
                        for children in nodeB.children :
                            children_b_id[children.representant] = children.id

                        if(children_a_id.__contains__(-1) or children_b_id.__contains__(-1) or children_b_id == [] or children_a_id == [] or children_a_id == [0]*len(children_a_id) or children_b_id == [0]*len(children_b_id)) : #On regarde si les deux noeuds ne sont pas dÃ©coupÃ©s
                            pass
                        else : #Si les deux noeuds sont dÃ©coupÃ©s
                            children_a = nodeA.children
                            children_b = nodeB.children
                            for (ca,cb) in zip(children_a,children_b) :#On relance la comparaison sur les enfants des noeuds
                                cadd = ca.clone()
                                cadd.parent = tree_compare.get_node(nodeAdd.id)
                                tree_compare.get_node(nodeAdd.id).children.append(cadd)
                                self.rec(ca,cb,cadd,tree_compare)

    def compare_tree(self,tree) :
        tree_compare = Tree(self.trainingSequence)

        self.rec(self.get_node_id(0),tree.get_node_id(0),self.get_node_id(0).clone(),tree_compare,isRoot = True)
        # for node in tree_compare.nodes :
        #     id = node.id
        #     print("ID : "+ str(id))
        #     print("RepTS: "+str(node.representant))
        #       printchildTS = [0] * (node.b ** 3)
        #       for children in node.children :
        #         printchildTS[children.representant] = children.id
        #
        #     print("CTS: "+ str(printchildTS))

        return tree_compare
        
    def compare_trunk(self, tree, returnRemainingNodes = False) : # ZP: find the common part(trunk) of 2 trees
        trunk = Tree(self.trainingSequence)

        nodes = self.rec_trunk(self.get_node_id(0), tree.get_node_id(0), self.get_node_id(0).clone(), trunk, isRoot = True)
        # for cat in nodes:
        #     print(cat)
        #     for node in nodes[cat]:
        #         print(node)
        if returnRemainingNodes:
            return [trunk, nodes]
        return (trunk)

    def rec_trunk(self, nodeA, nodeB, nodeAdd, tree_trunk, isRoot = False, remainingNodes = {'A':[], 'B':[]}) :
        if((nodeA.id != 0 and nodeB.id != 0) or isRoot) :
            b_a = nodeA.b
            b_b = nodeB.b
            tree_trunk.add_node_id(nodeAdd)

            tree_trunk.displayTree.add_graph_node(nodeAdd)

            if (b_a != -1 and b_b != -1) :
                if (b_a == b_b) :
                    # children_a_id = [0] * (b_a ** 3)
                    # for children in nodeA.children :
                    #     children_a_id[children.representant] = children.id
                    
                    # children_b_id = [0] * (b_b ** 3)
                    # for children in nodeB.children :
                    #     children_b_id[children.representant] = children.id
                    
                    # AisRemaining = False
                    # BisRemaining = False
                    # if children_a_id.__contains__(-1) or children_a_id == [] or children_a_id == [0]*len(children_a_id) :
                    #     AisRemaining = True
                    # if children_b_id.__contains__(-1) or children_b_id == [] or children_b_id == [0]*len(children_b_id) :
                    #     BisRemaining = True

                    # if AisRemaining or BisRemaining:
                    #     print(nodeA.children[0], nodeB.children[0])

                    # else : # if the 2 nodes are of the same children size
                    children_a = nodeA.children
                    children_b = nodeB.children
                    for (ca, cb) in zip(children_a, children_b) : 
                        if ca.representant == cb.representant : # add node only when it's the common node
                            cadd = ca.clone()
                            cadd.parent = tree_trunk.get_node(nodeAdd.id)
                            tree_trunk.get_node(nodeAdd.id).children.append(cadd)
                            self.rec_trunk(ca, cb, cadd, tree_trunk, False, remainingNodes)
                        else:
                            remainingNodes['A'].append(ca)
                            remainingNodes['B'].append(cb)

                else:
                    remainingNodes['A'].append(nodeA)
                    remainingNodes['B'].append(nodeB)

        return remainingNodes


    def getMaxDepth(self):
        return self.maxDepth()

    def maxDepth(self, root=None):
        if not root:
            # return 0
            root = self.get_root()
        if not root.children:
            return 1
        return 1 + max(self.maxDepth(child) for child in root.children)

    def get_root(self):
        return self.get_node_id(0)

    def get_node_id(self, id):
        for node in self.nodes:
            if node.id == id:
                return node 
        return None

    def add_node_id(self, node):
        node.id = len(self.nodes)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def getRelativeSize(self):
        size = 0
        nodesList = [self.get_node_id(0)] # Adding the root to the queue
        visited = []
        numberOfLeaves = len(self.get_node_id(0).children)

        while len(nodesList) > 0 :
            size += 1/(numberOfLeaves ** int(math.log2(nodesList[0].factor)))
            nodesList += nodesList[0].children
            visited.append(nodesList[0])
            del nodesList[0]
        return size

    def add_node_to_trunk(self, node, idParent = -1, isLeaf = True, includeChildren = False) :
        # It is possible to give an array (or dict) as parameter for "node"
        if isinstance(node, list):
            for n in node :
                self.add_node_to_trunk(n, idParent, isLeaf, includeChildren)

        elif isinstance(node, dict):
            for key in node:
                self.add_node_to_trunk(node[key], idParent, isLeaf, includeChildren)

        else:
            if node.parent:
                if idParent == -1 :
                    idParent = node.parent.id
                node.parent = self.get_node_id(idParent)

            newId = self.add_node_id(node)
            if node.parent:
                node.parent.isLeaf = False
                if idParent in self.leaves:
                    self.leaves.remove(idParent)
                if idParent in self.leaves_pending:
                    self.leaves_pending.remove(idParent)
                if node not in self.get_node_id(idParent).children:
                    self.get_node_id(idParent).children.append(node)
            if len(node.children) == 0:
                node.isLeaf = True

            if includeChildren:
                for child in node.children:
                    child.parent.id = newId
                    self.add_node_to_trunk(child, includeChildren=includeChildren)
        return self


    def normalizeLeaves(self) :
        self.leaves = []
        self.leaves_pending = []
        keptNodes = self.nodes 
        self.nodes = []
        id = 0
        while keptNodes != [] :
            node = keptNodes[0]
            node.id = id
            id += 1
            for child in node.children:
                if child not in keptNodes:
                    keptNodes.append(child)
            self.nodes.append(node)
            del keptNodes[0]

        for node in self.nodes:
            node.isLeaf = False

            allChildrenAreNone = True
            for child in node.children:
                if child.id > 0:
                    allChildrenAreNone = False
            if allChildrenAreNone :
                node.isLeaf = True
                self.leaves.append(node.id)
                self.leaves_pending.append(node.id)

        return self

    def display(self, suffix="tree"):
        self.normalizeLeaves()
        self.displayTree.reset()
        for n in self.nodes:
            self.displayTree.add_graph_node(n)
        self.displayTree.create_graph(suffix)


    def get_common_trunk(self, tree, returnRemainingNodes = False) : # HL : Getting common trunk, including leaves, returning difference of nodes
        self.normalizeLeaves()
        trunk = Tree(self.trainingSequence)
        remainingNodes = {'A':[], 'B':[]}
        self.rec_common_trunk(self.get_node_id(0), tree.get_node_id(0), self.get_node_id(0).clone(), trunk, isRoot = True, remainingNodes = remainingNodes)
        if returnRemainingNodes:
            return [trunk, remainingNodes]
        return (trunk)

    def rec_common_trunk(self, nodeA, nodeB, nodeAdd, tree_trunk, isRoot = False, remainingNodes = {'A':[], 'B':[]}) :
        if((nodeA.id != 0 and nodeB.id != 0) or isRoot) :
            b_a = nodeA.b
            b_b = nodeB.b
            newId = tree_trunk.add_node_id(nodeAdd)
            tree_trunk.displayTree.add_graph_node(nodeAdd)

            if (b_a != -1 and b_b != -1) :
                if (b_a == b_b) :
                    children_a = nodeA.children
                    children_b = nodeB.children
                    if [x.id for x in children_a] == [-1]*len(children_a) or [x.id for x in children_b] == [-1]*len(children_b):
                        remainingNodes['A'] += children_a
                        remainingNodes['B'] += children_b
                    else:
                        for (ca, cb) in zip(children_a, children_b) : 
                            if ca.representant == cb.representant : # add node only when it's the common node
                                cadd = ca.clone()
                                cadd.parent = tree_trunk.get_node(nodeAdd.id)
                                tree_trunk.get_node(nodeAdd.id).children.append(cadd)
                                self.rec_common_trunk(ca, cb, cadd, tree_trunk, False, remainingNodes)
                            else:
                                # If representant are not same, it's a difference
                                ca.parent.id = newId 
                                cb.parent.id = newId
                                remainingNodes['A'].append(ca)
                                remainingNodes['B'].append(cb)

                else:
                    # if the quantization is not the same, it's a difference
                    remainingNodes['A'].append(nodeA)
                    remainingNodes['B'].append(nodeB)

            # if b_a == -1:
            #     remainingNodes['A'].append(nodeA)
            # if b_b == -1:
            #     remainingNodes['B'].append(nodeB)

        return remainingNodes

    def countNodesRecursively(self):
        return self.get_root().countChildrenRecursively(includeSelf = True)

    def toPly(self, path):
        from tsplvq_to_ply import tsplvq_to_ply
        tsplvq_to_ply(self, path)
        return self
