import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
import random
import glob
import torchvision.models as models_res
import matplotlib.pyplot as plt
import autoencoder
import dataset  
import bce_fc_loss
import NNdataProcess
# #from skimage import io, transform
torch.cuda.empty_cache()



def train(model,n_epoch,optimizer,scheduler,device,data_dir,
          testPart,save_dir,type,is_hybird=False, hybird_dir=None):
    
    #MSE = nn.MSELoss()
    #loss = torch.nn.BCELoss()
    bceFCloss = bce_fc_loss.BCEFocalLoss(gamma=2, alpha=0.9)
    
    x_train, x_test = NNdataProcess.get_ae_data(testPart=testPart,data_dir=data_dir,is_return_dir=True)
    print(len(x_train),len(x_test))
    #exit(0)
    
    if hybird_dir:
        x2_train,x2_test = NNdataProcess.get_ae_data(testPart=testPart,data_dir=hybird_dir,is_return_dir=True)
        x_train, x_test = x_train+x2_train, x_test+x2_test
        random.shuffle(x_train)
        random.shuffle(x_test)
    
    print('start train AE')
    model.train()
    
    for epoch in range(3, n_epoch+1):   
        random.shuffle(x_train)
        
        for i in range(0,len(x_train),10):
            
            trainset = dataset.My_PLy_AE_Dataset(ply_dir = x_train[i:i+10], device=device,Normalize_= [],is_full_size=False,scale=0.3)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=False, num_workers=0)
            
            for batch_idx, (idex,x) in enumerate(trainloader):
               
                encoder,decoder = model(x)
                loss = bceFCloss(x, decoder)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
              
                if batch_idx % 1 == 0:
                    print('epoch {}: {}/{},loss: {}'.format(epoch, i, len(x_train),loss))

        #scheduler.step()
        #if epoch % 3 == 0 or epoch == n_epoch:
        torch.save(model.state_dict(), save_dir+'AutoEncoder-epode-' + str(epoch)+ '-'+ type + '.pkl')
    
    # model test 
    model.eval()
    with torch.no_grad():
        for i in range(len(x_test)-1):
            testset = dataset.My_PLy_AE_Dataset(ply_dir = x_test[i:i+1], device=device,Normalize_= [])
            testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
            for batch_idx, (idex,x) in enumerate(testloader):
                encoder,decoder = model(x)
                loss = bceFCloss(x, decoder)
                print('epoch {}: {}/{},Test-loss: {}'.format(epoch, 
                                    batch_idx*len(x_test),len(testloader.dataset),loss))
        

def creat_ply_by_AE(model,device,threshold,input_ply_dir,save_ply_dir):
    model.eval()
    MSE = nn.MSELoss()
    with torch.no_grad():
        
        testset = dataset.My_PLy_AE_Dataset(ply_dir =[input_ply_dir], device=device,Normalize_= None,)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
        for batch_idx, (idex,x) in enumerate(testloader):
            encoder,decoder = model(x)
            loss = MSE(x, decoder)
            print('Test-loss: {}'.format(loss))
            print(decoder.size())
            AE_decoder_out = torch.squeeze(decoder)
            print(AE_decoder_out.size())
            print(AE_decoder_out)
            tensor_to_ply(AE_decoder_out,threshold,save_ply_dir,scale_factor=None)
            
def tensor_to_ply(tensor,threshold,save_dir,scale_factor):
    return NNdataProcess.tensor_to_ply(tensor,threshold,save_dir,scale_factor)

def test_sacling_ply(input_ply_dir,save_ply_dir):
    device = torch.device('cpu')
    testset = dataset.My_PLy_AE_Dataset(ply_dir=[input_ply_dir], device=device,Normalize_= None,scale=0.3)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
    for batch_idx, (idex,x) in enumerate(testloader):
        x = torch.squeeze(x)
        print(x.size())
        tensor_to_ply(x,0.3,save_ply_dir,scale_factor=None)
    
def predicted_ply():
    AE_model_dir1 = model_path+ 'AutoEncoder-epode-12-3DCNN-AE-3(kernel-3-stride-2)-raw-PC.pkl'
    raw_ply = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/Ply/'
    rawPC_files = glob.glob(os.path.join(raw_ply, "*.ply"))
    
    AE1.load_state_dict(torch.load(AE_model_dir1))
    predicted_trunk_save_dir = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/predicted_trunk_by_nn/1051-AE3(kernal-3)-threadhold0.1.ply'
    
    #test_sacling_ply(raw_ply+'longdress_vox10_1051.ply',predicted_trunk_save_dir)
    
    creat_ply_by_AE(AE1,device,0.1,raw_ply+'longdress_vox10_1051.ply', predicted_trunk_save_dir)
   


if __name__=='__main__':
    raw_ply = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/Ply/'
    compressed_dir = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/Compressed_Ply/'
    traing_point_wise =  '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/training_dir_point_wise/'
    model_path = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/nn_part/model/'
    trunk_dir = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/trunk_ply/'
    
    
    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device('cpu')
    AE_model_dir1 = model_path+ 'AutoEncoder-epode-12-3DCNN-AE-3(kernel-3-stride-2)-raw-PC.pkl'
    AE1 = autoencoder.AE_3d_conv_3(hidden_size=16**3).to(device)
    AE1.load_state_dict(torch.load(AE_model_dir1))
    
    optimizer = optim.SGD(AE1.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.4)
    predicted_ply()
    exit(0)
 
    #print(AE1,sum([param.nelement() for param in AE1.parameters()]))
    #exit(0)
    
    train(model=AE1, n_epoch=50, optimizer=optimizer,scheduler=scheduler,
              device=device,data_dir = raw_ply,save_dir = model_path ,
              testPart=0.3,type='3DCNN-AE-3(kernel-3-stride-2)-scale0.3-raw-PC-bceFCloss',is_hybird=False,hybird_dir=None)


    # AE1 = autoencoder.AE_3d_conv(hidden_size=16**3).to(device)
    # optimizer = optim.SGD(AE1.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.4)

    # train(model=AE1, n_epoch=20, optimizer=optimizer,scheduler=scheduler,
    #           device=device,data_dir = trunk_dir,save_dir = model_path ,
    #           testPart=0.1,type='3DCNN-trunk-PC')

    
    