# -*-coding:Latin-1 -*
"""
Voronoi cell keeping information on the node and children voronoi cell
"""

from codes.VoronoiCell import VoronoiCell
class VoronoiCellNode(VoronoiCell):     # Inherited from VoronoiCell class
    def __init__(self, representant, dimension, children, node):    # 2 more parameters : children and node
        VoronoiCell.__init__(self, representant, dimension)
        self.children = children
        self.node = node

    def scale(self, factor):
        self.representant *= factor     # scale the representation
        self.dimension *= factor    # scale the dimension
        for child in self.children:     # scale the representation and dimension in each child
            child.scale(factor)

    def translate(self, vec):
        self.representant += vec    # add vec to each vector in the representation
        for child in self.children:     # do it to each child among children
            child.translate(vec)

    def __len__(self):      # define a special method
        len = 1
        for child in self.children:
            len += len(child)
        return len      # return the number of children + 1 (this number include the corresponding parent)

    def retrieve_all(self):
        elements = [self]   # define an array named element
        for child in self.children:     # retrieve all the children VoronoiCell
            elements.extend(child.retrieve_all())
        return elements

    def print_geometric_centers (self,pro) :
        for voronoi_cell_node in self.children:
            if voronoi_cell_node.children == []:
                pro.j = pro.j + 1
                print("geometric center of voronoi leaf = ", voronoi_cell_node.representant, "node associated = ", voronoi_cell_node.node.id)
            else:
                voronoi_cell_node.print_geometric_centers(pro)
