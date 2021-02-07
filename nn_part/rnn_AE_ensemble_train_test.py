import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from plot_rnn import PlotterRNN
from rnn import RNN
import numpy as np
import dataset 
from codes import NNdataProcess


class TrainRNN:
    def __init__(self, rnn, device,batch_size=1, learning_rate=0.01, epoch=10):
        
        self.rnn = rnn
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
        
    def train_test(self, testPart=0.1,data_dir='',type=None,save_dir=''):                    
        x_trains, y_trains, x_tests, y_tests = self.get_training_data(testPart=testPart,data_dir=data_dir)
        
        MSE = nn.MSELoss()
        model = self.rnn
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        print('start train RNN')

        model.train()
              
        for epoch in range(self.epoch):
            epoch_loss = 0
            for i in range(len(x_trains)):
                x = x_trains[i]
                y = y_trains[i]
                x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
                y = torch.from_numpy(y).type(torch.FloatTensor).to(self.device)
                
           
                hidden = model.init_hidden()
                batch_x = Variable(x)
                batch_y = Variable(y)
                output, hidden = model(batch_x, hidden)
                
                loss = MSE(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss
#                 if i % 10 == 0:
#                     print('epoch {}: {}/{},loss: {}'.format(epoch, i, len(x_trains),loss))
            if epoch % 1 == 0:
                print('epoch {}: avg_loss: {}'.format(epoch,epoch_loss/len(x_trains)))
                    
        torch.save(model.state_dict(), save_dir+'RNN-GRU-layer:'+'epode-' + str(epoch)+ '-'+ type + '.pkl')
        
        
        print('start to test')
        model.eval()
        with torch.no_grad():
            for i in range(len(x_tests)):
                x = x_tests[i]
                y = y_tests[i]
                x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
                y = torch.from_numpy(y).type(torch.FloatTensor).to(self.device)
                hidden = model.init_hidden()
                batch_x = Variable(x)
                batch_y = Variable(y)
                output, hidden = model(batch_x, hidden)
                loss = MSE(output, batch_y)
               
                print('Number: {}/{},loss: {}'.format( i, len(x_tests),loss))
                   
    def get_training_data(self,testPart,data_dir):
        return NNdataProcess.get_data(testPart=testPart,data_dir=data_dir,is_return_dir=False)
 
if __name__=='__main__':
    rnn_train_tensor = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/seq/longdress/rnn_train_sequence_tensor_by_AE/'
    model_path = '/datavico/home/liw/my_code/Pred-INFO5-2017-2018-TSLVQ-1/nn_part/model/'

    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device('cpu')

    rnn_model = RNN(input_size=4096, hidden_size=512, output_size=4096, n_layers=3)
    train = TrainRNN(rnn_model,device,batch_size=1,learning_rate=0.005,epoch=50)
    train.train_test(testPart=0.05,data_dir=rnn_train_tensor,type='n_layers:'+str(3)+'-rawPC_AE:4096',save_dir=model_path)
