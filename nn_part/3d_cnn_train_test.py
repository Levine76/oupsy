import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
#import pandas as pd
import torchvision.models as models_res
import matplotlib.pyplot as plt

import dataset  
import 3d_cnn_model  
import dataProcess
# #from skimage import io, transform

torch.cuda.empty_cache()


def Training_Point_Cloud_Compress_by_using_CNN(model, device, train_data_dir,test_data_dir, testPart, optimizer,scheduler,n_epoch,type):
    model.train()
    MSE = nn.MSELoss()
    x_train, y_train, x_test, y_test = dataProcess.get_data(testPart=testPart,data_dir=train_data_dir,is_return_dir=True)
    
    for epoch in range(1, n_epoch+1):   
        for i in range(len(x_train)-1):
            trainset = dataset.My_PLy_Dataset(ply_dir = x_train[i:i+1],lable_dir = y_train[i:i+1], device=device,Normalize_= [])
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
            for batch_idx, (idex,x,y) in enumerate(trainloader):
                out = model(x)       
                optimizer.zero_grad()
                loss = MSE(out, y)
                loss.backward()
                optimizer.step()
                if batch_idx % 1 == 0:
                    print('epoch {}: {}/{},loss: {}'.format(epoch, batch_idx*len(x_train), len(trainloader.dataset),loss))
            scheduler.step()
    torch.save(model.state_dict(), 'model/PCCC-epode-' + str(epoch)+ '-'+type + '.pkl')
    return model_test(model,device,test_data_dir,testPart)
    
       
def model_test(model,device,test_data_dir,testPart):
    x_train, y_train, x_test, y_test = dataProcess.get_data(testPart=testPart)
    model.eval()
    MSE = nn.MSELoss()
    out_loss_list = []
    with torch.no_grad():
        for i in range(len(x_test)-1):
            testset = dataset.My_PLy_Dataset(ply_dir = x_test[i:i+1],lable_dir = y_test[i:i+1], device=device,Normalize_= [])
            testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
            
            for batch_idx, (idex,x,y) in enumerate(testloader):
                out = model(x)  
                loss = MSE(out, y)
                out_loss_list.append([loss,out])
                print([loss,out])
    return np.array(out_loss_list)

             
if __name__=='__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device('cpu')


    file_dir = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/Ply/'

    traing_point_wise = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/training_dir_point_wise/'

    mode_name = 'model/PCCC-epode-50.pkl'

    net1 = My_3D_CNN_Model.My_3d_cnn_model(out_feature=4096) 
    net1 = net1.to(device)
    #net1.load_state_dict(torch.load(mode_name))

    #My_3D_CNN_Model.model_test(model=net1,device=device,data_dir=file_dir,testPart=0.8)    


    optimizer = optim.SGD(net1.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.4)

    My_3D_CNN_Model.Training_Point_Cloud_Compress_by_using_CNN(
            model=net1,device=device,train_data_dir = traing_point_wise,
        test_data_dir = file_dir,
        testPart=0.2,optimizer=optimizer,scheduler=scheduler,n_epoch=100,type='TrainSet_54') 
