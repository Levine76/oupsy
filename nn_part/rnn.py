import torch
import torch.nn as nn
from torch.autograd import Variable
import autoencoder

class RNN(nn.Module):
    def __init__(self, input_size=4096, hidden_size=512, output_size=4096, n_layers=3):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        
        self.gru = nn.GRU(input_size, hidden_size, n_layers)    
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, input, hidden):
        # input_size = (timeSize,batchSize,featureSize)
        print(input.size())
        output, hidden = self.gru(input.view(-1, 1, 4096), hidden)
        print(output.size())
        output = self.fc(output)
        #only need the last item of time sequence 
        output = output[-1,:,:]
        print(output.size())
        
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
    



class RNN_AE(nn.Module):
    def __init__(self, input_size=4096, hidden_size=512, output_size=4096, n_layers=3):
        super(RNN_AE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.encoder = autoencoder.Encoder_conv3D_4(input_size)
        self.gru = nn.GRU(input_size, hidden_size, n_layers)    
        self.fc = nn.Linear(hidden_size, output_size)
        self.decoder = autoencoder.Dncoder_conv3D_4(output_size)

    def forward(self, input, hidden):
        # input_size = (timeSize,batchSize,featureSize)
       
        #output = self.encoder(input)
        output = torch.zeros(1,self.input_size)
        print('encoder part')
        for i in range(len(input)):
            if i == 0:
                output = self.encoder(input[i:i+1])
            else:
                output = torch.cat([output,self.encoder(input[i:i+1])],0)
            
        output, hidden = self.gru(output.view(-1, 1, 4096), hidden)
        output = self.fc(output)
        
        #only need the last item of time sequence 
        output = output[-1,:,:]
        
        print('decoder part')
        output = self.decoder(output)
        #hidden = self.decoder(hidden)
        
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))