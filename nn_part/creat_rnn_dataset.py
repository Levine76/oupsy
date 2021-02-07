import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
import random
import pickle
import torchvision.models as models_res
import matplotlib.pyplot as plt
import autoencoder
import dataset  
import NNdataProcess
# #from skimage import io, transform
torch.cuda.empty_cache()


def get_hidden_tensor(model,device,data_dir,save_dir):
    
   
    raw_pc_list,trunk_pc, _,_ = NNdataProcess.get_data(testPart=0,data_dir=data_dir,is_return_dir=True)
    
    print('start eval AE') 
    
    model.eval()
    with torch.no_grad():
        MSE = nn.MSELoss()
        for i in range(len(raw_pc_list)):
            feature_vector = []
            trunk_vector = []
           
            pcSet = dataset.My_PLy_AE_Dataset(ply_dir = raw_pc_list[i],device=device, Normalize_= [])
            pcloader = torch.utils.data.DataLoader(pcSet, batch_size=1, shuffle=False, num_workers=0)
            for batch_idx, (idex,x) in enumerate(pcloader):
                encoder,decoder = model(x)
                feature_vector.append(encoder.numpy().tolist()[0])
                loss = MSE(x, decoder)
                print('Test-loss: {}'.format(loss))
            
            trunkSet = dataset.My_PLy_AE_Dataset(ply_dir = [trunk_pc[i]],device=device, Normalize_= [])
            trunkloader = torch.utils.data.DataLoader(trunkSet, batch_size=1, shuffle=False, num_workers=0)
            for batch_idx, (idex,x) in enumerate(trunkloader):
                encoder,decoder = model(x)
                trunk_vector.append(encoder.numpy().tolist()[0])
                loss = MSE(x, decoder)
                print('Test-loss: {}'.format(loss))
            
            # print(len(feature_vector),len(feature_vector[0]))
            # print(len(trunk_vector),len(trunk_vector[0]))
            # exit(0)
            if save_dir:
                file_dump = open(save_dir+'AE-tensor(RawPC-'+'TrunkPC)_'+str(i) +".obj",'wb')
                pickle.dump([[feature_vector],trunk_vector], file_dump)
                file_dump.close()

if __name__=='__main__':
    
    model_path = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/nn_part/model/'
    rnn_train_dir = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/rnn_train_sequence/'
    rnn_train_tensor = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/rnn_train_sequence_tensor_by_AE/'

    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device('cpu')
    AE = autoencoder.AE_3d_conv(hidden_size=16**3).to(device)   
    AE.load_state_dict(torch.load(model_path+ 'AutoEncoder-epode-20-3DCNN-raw-PC.pkl' ))

    get_hidden_tensor(AE,device, data_dir=rnn_train_dir ,save_dir=rnn_train_tensor)
        