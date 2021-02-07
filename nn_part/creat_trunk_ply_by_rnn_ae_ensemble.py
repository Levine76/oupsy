import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from rnn import RNN
import os
import random
import glob
import numpy as np
import autoencoder
import dataset 
import NNdataProcess
import glob


def get_rnn_out(model,rawPc_tensor_seq):
    model.eval()
    with torch.no_grad():
        x = rawPc_tensor_seq
        hidden = model.init_hidden()
        batch_x = Variable(x)
        output, hidden = model(batch_x, hidden)
        return output
        
def get_AE_decoder_out(model,rnn_out_tensor):
    model.eval()
    with torch.no_grad():
        return model.decoder(rnn_out_tensor)

def get_rnn_in_featur(model,device,rawPc_ply_dir_list):
    model.eval()
    with torch.no_grad():
        feature_vector = torch.zeros(1,4096).to(device)
        for i in range(len(rawPc_ply_dir_list)):
            pcSet = dataset.My_PLy_AE_Dataset(ply_dir = [rawPc_ply_dir_list[i]],device=device, Normalize_= [])
            pcloader = torch.utils.data.DataLoader(pcSet, batch_size=1, shuffle=False, num_workers=0)
            for batch_idx, (idex,x) in enumerate(pcloader):
                
                encoder,decoder = model(x)
                if i == 0:
                    feature_vector = encoder
                else:
                    feature_vector = torch.cat((feature_vector, encoder), 0)
               
    return feature_vector

def tensor_to_ply(tensor,threshold,save_dir):
    return NNdataProcess.tensor_to_ply(tensor,threshold,save_dir)

def creat_trunk_ply_by_nn(AE_model_dir1,AE_model_dir2,RNN_model_dir,rawPc_ply_dir_list,device,threshold,save_dir):
    
    rnn_model = RNN().to(device)
    rnn_model.load_state_dict(torch.load(RNN_model_dir))
    
    AE_model1 = autoencoder.AE_3d_conv().to(device)
    AE_model1.load_state_dict(torch.load(AE_model_dir1))
    
    rnn_in_feature = get_rnn_in_featur(AE_model1,device,rawPc_ply_dir_list)
    rnn_out = get_rnn_out(rnn_model,rnn_in_feature)
    
#     AE_model2 = autoencoder.AE_3d_conv().to(device)
#     AE_model2.load_state_dict(torch.load(AE_model_dir2))
    
    AE_decoder_out = get_AE_decoder_out(AE_model1,rnn_out)
    #point_counts = len(AE_decoder_out[AE_decoder_out>0.2])
    AE_decoder_out = torch.squeeze(AE_decoder_out)
    
    print('try to save predicted point cloud(.ply) by NN ')
    tensor_to_ply(AE_decoder_out,threshold,save_dir)
    


  
if __name__=='__main__':
    model_path = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/nn_part/model/'
    raw_ply = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/Ply/'
    predicted_trunk_save_dir = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/predicted_trunk_by_nn/A12.ply'
    
    AE_model_dir1 = model_path+ 'AutoEncoder-epode-20-3DCNN-raw-PC.pkl'
    AE_model_dir2 = model_path+ 'AutoEncoder-epode-20-3DCNN-trunk-PC.pkl'
    RNN_model_dir = model_path+ 'RNN-GRU-layer:epode-49-n_layers:3-rawPC_AE:4096.pkl'
    
    rawPC_files = glob.glob(os.path.join(raw_ply, "*.ply"))
    rawPC_files.sort()   
    rawPc_ply_dir_list = rawPC_files[0:3]
    
    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device('cpu')
    
    creat_trunk_ply_by_nn(AE_model_dir1,AE_model_dir2,RNN_model_dir,
                          rawPc_ply_dir_list,device,threshold=0.2,save_dir=predicted_trunk_save_dir)