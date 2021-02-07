import glob
import os
import open3d
from personal_parameters.myConfig import config

#from codes import ObjLoader
path=config.pathToLongdress2Compressed_Ply
treefiles = glob.glob(os.path.join(path, "*.ply"))
N=2000
for file in treefiles:
    pcd=open3d.io.read_point_cloud (file)

voxel_grid = open3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                              voxel_size=0.8)
open3d.visualization.draw_geometries([voxel_grid])

