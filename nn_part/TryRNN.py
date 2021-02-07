import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
#from plot_rnn import PlotterRNN
from nn_part.rnn import RNN_AE
import numpy as np
#from nn_part import dataset
#from codes import NNdataProcess
#import random
from nn_part import bce_fc_loss
from personal_parameters.myConfig import config

class TryRNN:
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

    def run(self, model_path, dir_in, dir_out):
        model = self.rnn
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        model = torch.load(model_path)
        #model.load_state_dict(modelRNN['model_state_dict'])
        #optimizer.load_state_dict(modelRNN['optimizer_state_dict'])

        model.eval()





if __name__=='__main__':

    model_path = config.pathToLongdressRNNModel + "\RNN-GRU-layer_epode_146_.pkl"
    dir_in = config.pathToLongdressPly
    dir_out = config.pathToLongdressRNNPly

    use_cuda = torch.cuda.is_available()
    device = torch.device('cpu')

    rnn_model = RNN_AE(input_size=4096, hidden_size=512, output_size=4096, n_layers=3)

    RNN_to_try = TryRNN(rnn_model,device,batch_size=1,learning_rate=0.01,epoch=500,loss_func='MSE')
    RNN_to_try.run(model_path, dir_in, dir_out)

