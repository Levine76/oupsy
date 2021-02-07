import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from plot_rnn import PlotterRNN
from nn_part.rnn import RNN_AE
import numpy as np
from nn_part import dataset
from codes import NNdataProcess
import random
from nn_part import bce_fc_loss
from personal_parameters.myConfig import config

class TrainRNN_AE:
    def __init__(self, rnn, device,batch_size=1, learning_rate=0.01, epoch=10,loss_func=''):
        
        self.rnn = rnn
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        
        if loss_func == 'MSE':
            self.Loss = nn.MSELoss() 
        elif loss_func == 'BCEFL':
            self.Loss = bce_fc_loss.BCEFocalLoss(gamma=2, alpha=0.9)
            
        
    def train_test(self, testPart=0.1,data_dir='',type=None,save_dir=''):   
        
        x_trains, y_trains, x_tests, y_tests = self.get_training_data(testPart=testPart,data_dir=data_dir)    
        model = self.rnn
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        print('start train RNN')

        model.train()
        for epoch in range(0,self.epoch):
            for i in range(len(x_trains)):
                trainset = dataset.My_PLy_RNN_AE_Dataset(ply_dir = x_trains[i],lable_dir = [y_trains[i]],device=device,Normalize_= [],is_full_size=False,scale=0.3)
                
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=False, num_workers=0)
                for batch_idx, (idex,x,y) in enumerate(trainloader):
                    print(batch_idx,len(x))
                    if len(x) == 1 and batch_idx != 0:
                        continue
                    
                    hidden = model.init_hidden()
                    batch_x = Variable(x)
                    batch_y = Variable(y)
                    output, hidden = model(batch_x, hidden)
                    loss = self.Loss(output[0],batch_y[0])
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print('epoch {}: {}/{},loss: {}'.format(epoch, i, len(x_trains), loss))

                    #if i % 10 == 0:
                         #print('epoch {}: {}/{},loss: {}'.format(epoch, i, len(x_trains),loss))
                    
            #torch.save(model.state_dict(), save_dir+'\RNN-GRU-layer_'+'epode_' + str(epoch)+ '_'+ '.pt')
        torch.save(model.state_dict(), save_dir + '\RNN-GRU-layer_' + 'epode_' + str(epoch)+ '.pt')
            #+ type
        
        
        print('start to test')
        model.eval()
        with torch.no_grad():
            for i in range(len(x_tests)-1):
                testset = dataset.My_PLy_RNN_AE_Dataset(ply_dir = x_trains[i],lable_dir = [y_tests[i]], device=device,Normalize_= [],is_full_size=False,scale=0.3)
                testloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
                for batch_idx, (idex,x,y) in enumerate(testloader):
        
                    hidden = model.init_hidden()
                    batch_x = Variable(x)
                    batch_y = Variable(y)
                    output, hidden = model(batch_x, hidden)
                    
                    loss = MSE(output, batch_y)
                    if i % 1 == 0:
                         print('epoch {}: {}/{},loss: {}'.format(epoch, i, len(x_trains),loss))
                                 
    def get_training_data(self,testPart,data_dir):
        return NNdataProcess.get_data(testPart=testPart,data_dir=data_dir,is_return_dir=True)
    
    def tensor_to_ply(tensor,threshold,save_dir):
        return NNdataProcess.tensor_to_ply_(tensor,threshold,save_dir,scale_factor=0.3,Original_dimension=1024)


     
if __name__=='__main__':
    
    model_path = config.pathToLongdressRNNModel
    rnn_train_dir = config.pathToLongdressRNNTrain

    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device('cpu')
    
    #rnn_AE_model_dir1 = model_path + 'RNN-GRU-layer:epode-56-n_layers:3-rawPC_AE(embed):4096.pkl'
    rnn_model = RNN_AE(input_size=4096, hidden_size=512, output_size=4096, n_layers=3)
    #rnn_model.load_state_dict(torch.load(rnn_AE_model_dir1))
    train = TrainRNN_AE(rnn_model,device,batch_size=1,learning_rate=0.01,epoch=500,loss_func='MSE')
    train.train_test(testPart=0.2,data_dir=rnn_train_dir,type='n_layers:'+str(3)+'-rawPC_AE(embed):4096-MSE-linear',save_dir=model_path)
