import torch
import torch.utils.data
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

import sys
#sys.path.append(r"/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1")
from codes import ObjLoader
from codes import NNdataProcess

#courtesy of https://github.com/meder411/PointNet-PyTorch.git

class My_PLy_AE_Dataset(Dataset):
    def __init__(self,ply_dir, device,Normalize_=None,is_full_size=False,scale=None):
        
        ply_numpy_list = []   
        ply_numpy_list_lable = []
        for fil in ply_dir:

            pc = ObjLoader.load_point_cloud(fil)
            pc.vertices = pc.vertices.astype('float16') 
          
            #pc.colors = pc.colors.astype('float16')
            if scale:
                ply_numpy_list.append(NNdataProcess.transform_scaling(pc.vertices,Original_dimension=1024,scale_factor=scale))
            else:
                ply_numpy_list.append(NNdataProcess.transform(pc.vertices,is_full_size=is_full_size))  #,pc.colors))
            
        self.Tensor_ply_list = torch.from_numpy(np.array(ply_numpy_list)).type(torch.FloatTensor).to(device)
      
        
    def __getitem__(self, idx):
        sample = self.Tensor_ply_list[idx]   
        return idx, sample
    
    def getdata(self):
        return self.Tensor_ply_list
    
    def __len__(self):
        return len(self.Tensor_ply_list)
    
    
    
    
class My_PLy_Dataset(Dataset):
    def __init__(self,ply_dir, lable_dir, device,Normalize_=None,is_full_size=False,scale=None):
        
        ply_numpy_list = []   
        ply_numpy_list_lable = []
        for fil in ply_dir:
            pc = ObjLoader.load_point_cloud(fil)
            pc.vertices = pc.vertices.astype('float16') 
            #pc.colors = pc.colors.astype('float16')
            if scale:
                ply_numpy_list.append(NNdataProcess.transform_scaling(pc.vertices,Original_dimension=1024,scale_factor=scale))
            else:
                ply_numpy_list.append(NNdataProcess.transform(pc.vertices))  #,pc.colors))
        
        self.Tensor_ply_list = torch.from_numpy(np.array(ply_numpy_list)).type(torch.FloatTensor).to(device)
        
        for fil in lable_dir:
            pc_lable = ObjLoader.load_point_cloud(fil)
            pc_lable.vertices = pc_lable.vertices.astype('float16') 
            #pc.colors = pc.colors.astype('float16')
            if scale:
                ply_numpy_list_lable.append(NNdataProcess.transform_scaling(pc_lable.vertices,Original_dimension=1024,scale_factor=scale)) 
            else:
                ply_numpy_list_lable.append(NNdataProcess.transform(pc_lable.vertices))  #,pc.colors))
            
        
        self.Tensor_ply_list_lable = torch.from_numpy(np.array(ply_numpy_list_lable)).type(torch.FloatTensor).to(device)
            
  
        
    def __getitem__(self, idx):
        sample = self.Tensor_ply_list[idx]      
        lable = self.Tensor_ply_list_lable[idx]
        return idx, sample,lable

    def __len__(self):
        return len(self.Tensor_ply_list)
   

class My_PLy_RNN_AE_Dataset(Dataset):
    def __init__(self,ply_dir, lable_dir, device,Normalize_=None,is_full_size=False,scale=None):
        
        ply_numpy_list = []   
        ply_numpy_list_lable = []
        for fil in ply_dir:

            pc = ObjLoader.load_point_cloud(l)
            pc.vertices = pc.vertices.astype('float16')

            #pc.colors = pc.colors.astype('float16')
            if scale:
                ply_numpy_list.append(NNdataProcess.transform_scaling(pc.vertices,Original_dimension=1024,scale_factor=scale))
            else:
                ply_numpy_list.append(NNdataProcess.transform(pc.vertices))  #,pc.colors))
        
        self.Tensor_ply_list = torch.from_numpy(np.array(ply_numpy_list)).type(torch.FloatTensor).to(device)
        
        for fil in lable_dir:
            pc_lable = ObjLoader.load_point_cloud(fil)
            pc_lable.vertices = pc_lable.vertices.astype('float16')
            #pc.colors = pc.colors.astype('float16')
            if scale:
                ply_numpy_list_lable.append(NNdataProcess.transform_scaling(pc_lable.vertices,Original_dimension=1024,scale_factor=scale))
            else:
                ply_numpy_list_lable.append(NNdataProcess.transform(pc_lable.vertices))  #,pc.colors))

        
        self.Tensor_ply_list_lable = torch.from_numpy(np.array(ply_numpy_list_lable)).type(torch.FloatTensor).to(device)
            
  
        
    def __getitem__(self, idx):
        sample = self.Tensor_ply_list[idx]      
        lable = self.Tensor_ply_list_lable[0]
        #print(lable.size())
        return idx, sample,lable

    def __len__(self):
        return len(self.Tensor_ply_list)