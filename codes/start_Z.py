import sys
import os

from codes import ObjLoader
from codes import TrainingSequence
from codes import Quantization
from codes import Tree
from codes import QuantizationMethodBase
from codes import Probability
from personal_parameters.myConfig import config

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import open3d
import os.path as osp
import pickle

## loading points cloud
def Compressed_PLY(file, loader_dir,save_dir,rateMax = 10000):
    pcd = ObjLoader.load_point_cloud(file) #opens an object of the ply point cloud in the loader_dir file
    pcd_protected = ObjLoader.load_point_cloud(file) # protecting the original points cloud information into pcd_protected, so the modifications on pcd don't make any influnce on pcd_protected

    ## Creating training sequence
    ts = TrainingSequence.TrainingSequence()
    ts_protected = TrainingSequence.TrainingSequence()
    ts.init(pcd)
    ts.compute_normals()
    ts_protected.init(pcd_protected)
    ts_protected.compute_normals()

    ## Creating quantization instance
    qt = Quantization.Quantization()
    qt_protected = Quantization.Quantization()
    qt.add_method(QuantizationMethodBase.QuantizationMethodBase(2))
    qt.add_method(QuantizationMethodBase.QuantizationMethodBase(3))
    qt_protected.add_method(QuantizationMethodBase.QuantizationMethodBase(2))

    ## Entropic code
    pro = Probability.Probability()

    ## Creating tree
    tree = Tree.Tree(ts)
    tree.create_root()
    tree_protected = Tree.Tree(ts_protected)

    ## Setting maximun rate score


    ## Quantization Loop
    while tree.rate < rateMax:
        qt.quantize(tree,pro)
        #print(str(tree.size()) + ", " + str(tree.rate) + ", " + str(tree.distortion))
        #SP: visualisation intermédiaire ici


    root = tree.get_node(0)     # Getting the root, the root includes all the information of the tree
    voronoiCellNode = root.get_voronoiCellNode(qt, np.array([0.0, 0.0, 0.0]))   # Getting root voronoi cell node

    points_plot = np.zeros((1, 6))
    points_plot[0][3] = 225
    points_plot[0][4] = 225
    points_plot[0][5] = 225

    def calcul(node, points_plot):
        if node.node.isLeaf:
            inputs = ts_protected.retrieve(node.node.inputs)#
            center = inputs.mean(0)
            mean_color = ts_protected.mean_color(node.node.inputs)
            points_d = np.zeros((1, 6))#channel
            points_d[0][0] = center[0]#information couleur , (donnée de la couleur R pour 1 point)
            points_d[0][1] = center[1]
            points_d[0][2] = center[2]
            points_d[0][3] = mean_color[0]
            points_d[0][4] = mean_color[1]
            points_d[0][5] = mean_color[2]
            points_plot = np.concatenate((points_plot, points_d), axis=0)

        for child in node.children:
            points_plot = calcul(child, points_plot)

        return points_plot

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


    points_plot = calcul(voronoiCellNode, points_plot)
    write_ply(points_plot, save_dir)

def creationDataTraining_PLY_Y(dirs = ["data/longdress/Ply/"],index = ["longdress" ],) :
    analyse = {}
    compt = 0
    compt_ind = 0
    #dirs = ["../data/longdress/Ply/"]   #,"seq/loot/Ply/","seq/redandblack/Ply/","seq/soldier/Ply/"]
    #index = ["longdress" ] #,"loot","redandblack","soldier"]
    for dir in dirs :
        files = os.listdir(dir)
        files.sort()
        for nb in range(len(files)) :
            #if files[nb][0] == '.': continue
            save_file = "data/longdress2/Compressed_Ply/longdress_compressed_"+files[nb]
            print("compressing file : ", dir+files[nb])
            Compressed_PLY(dir+files[nb],save_file, 10000)
            #print("creating training sequence for : ", files[nb])
            #file_dump = open("../trainingSeq/train_"+index[compt_ind]+str(compt)+".obj",'wb')
            #pickle.dump([dir+files[nb],save_file], file_dump)
            #file_dump.close()
            #print("created : training sequence train_"+ index[compt_ind]+str(compt)+".obj")
            compt+=1
        compt_ind+=1

#compression en ply sans créer de training sequence
#def CompressSrcToPLY(src_dir = ["data/longdress/Ply/"],save_dir = "data/longdress/Compressed_Ply/",rateMax = 10000) :
def CompressSrcToPLY(file, src_folder ,save_dir, rateMax) :
    analyse = {}
    #src_dir = config.pathToLongdressPly
    #save_dir = config.pathToLongdressTrunk
    src_dir = [src_folder]
    for dir in src_dir :
        files = os.listdir(dir)
        files.sort()
        for nb in range(len(files)) :
            #if files[nb][0] == '.': continue
            save_file = save_dir + "\compressed_" + files[nb]
            print("compressing file : ", dir+files[nb])
            Compressed_PLY(file, dir+files[nb],save_file,rateMax)
