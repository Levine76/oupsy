import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from rnn import RNN_AE
import os
import random
import glob
import numpy as np
import autoencoder
import dataset 
from codes import NNdataProcess
import glob
from personal_parameters.myConfig import config


def get_rnn_out(model,rawPc_dir_seq):

    model.eval()
    with torch.no_grad():
        testset = dataset.My_PLy_AE_Dataset(ply_dir = rawPc_dir_seq, device=device,Normalize_= [],is_full_size=False,scale=1.0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        for batch_idx, (idex,x) in enumerate(testloader):
        
            hidden = model.init_hidden()
            batch_x = Variable(x)

            output = batch_x
            output, hidden = model(batch_x, hidden)
            output = torch.squeeze(output)
    
    return output


def tensor_to_ply(tensor,threshold,save_dir,scale_factor=None,Original_dimension=1024):
    return NNdataProcess.tensor_to_ply(tensor,threshold,save_dir,scale_factor=scale_factor,Original_dimension=Original_dimension)

def creat_trunk_ply_by_nn(RNN_AE_model_dir,rawPc_ply_dir_list,device,threshold,save_dir):
    
    rnn_model = RNN_AE().to(device)
    rnn_model.load_state_dict(torch.load(RNN_AE_model_dir))
    
    rnn_out = get_rnn_out(rnn_model,rawPc_ply_dir_list)
    
    print('try to save predicted point cloud(.ply) by RNN_AE')
    tensor_to_ply(rnn_out,threshold,save_dir)
    


  
if __name__=='__main__':
    model_path = config.pathToLongdressRNNModel
    raw_ply = config.pathToLongdressPly
    predicted_trunk_save_dir = config.pathToLongdress + '/predicted_trunk_by_nn/1051-sale1.ply'
  
    #RNN_AE_model_dir = model_path + '/RNN-GRU-layer_epode_146_.pkl'
    RNN_AE_model_dir = model_path + '/RNN-GRU-layer_epode_4_BCEFL.pt'

    rawPC_files = glob.glob(os.path.join(raw_ply, "*.ply"))
    rawPC_files.sort()   
    rawPc_ply_dir_list = rawPC_files[0:1]
    
    print(rawPc_ply_dir_list)

    use_cuda = torch.cuda.is_available()
    if(use_cuda):
        device = torch.device('cuda:0')
        print("utilisation de cuda")
    else:
        device = torch.device('cpu')
        print("utilisation de cpu")


    creat_trunk_ply_by_nn(RNN_AE_model_dir,rawPc_ply_dir_list,device,threshold=0.1,save_dir=predicted_trunk_save_dir)