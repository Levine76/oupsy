import torch 
import torch.nn as nn
import torch.nn.functional as F
torch.cuda.empty_cache()

class Decoder_linear(nn.Module):
    def __init__(self, in_features):
        super().__init__()        
        
        self.fc1 = nn.Sequential(nn.Linear(in_features,64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64,64), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(64,512**3), nn.ReLU(inplace=True))# not works on this server too 
        #self.fc4 = nn.Sequential(nn.Linear(256,512**3), nn.ReLU(inplace=True))
        
        # 64*512**3 = 64G
        
    def forward(self, z):
        
        out = self.fc1(z)
        print(out.size())
        out = self.fc2(out)
        print(out.size())
        out = self.fc3(out)
        print(out.size())
        output = output.view(-1, 3, 512)
        exit(0)
        return output


class Encoder_conv1D(nn.Module):
    def __init__(self, out_features):
        super().__init__()

        self.z_size = out_features 
        self.pc_encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=self.z_size, kernel_size=1))

            
    def forward(self, x):
        output = self.pc_encoder_conv(x)
        
        #out = out.view(out.size(0),-1)
        output = torch.sigmoid(output)
        
        output = output.max(dim=2)[0]
        # output = self.pc_encoder_fc(output)
        return output
    

class Encoder_conv3D(nn.Module):
    def __init__(self, out_features):
        super().__init__()

        #self.input_conv1 = nn.Sequential(nn.Conv3d(in_channels=1,out_channels=16,kernel_size=2, stride=3),nn.ELU() ) 
        self.input_conv2 = nn.Sequential(nn.Conv3d(in_channels=1,out_channels=32,kernel_size=2, stride=3),nn.ELU() ) 
        self.input_conv3 = nn.Sequential(nn.Conv3d(in_channels=32,out_channels=32,kernel_size=2, stride=3),nn.ELU(), nn.BatchNorm3d(32))
        self.input_conv4 = nn.Sequential(nn.Conv3d(in_channels=32,out_channels=64,kernel_size=2, stride=3),nn.ELU(),nn.BatchNorm3d(64))
        self.input_conv5 = nn.Sequential(nn.Conv3d(in_channels=64,out_channels=64,kernel_size=2, stride=3),nn.ELU(),nn.BatchNorm3d(64))
       
        self.fc1 = nn.Linear(13824,4096) # 16**3
        #self.fc2 = nn.Linear(512,out_features)
      
    def forward(self,x):
        #out = self.input_conv1(x)
        print(x.size())
        out = self.input_conv2(x) 
        
        print(out.size())
        out = self.input_conv3(out)
        print(out.size())
        out = self.input_conv4(out)
        print(out.size())
        out = self.input_conv5(out)
        print(out.size())
        out = out.view(out.size(0),-1)
        print(out.size())
        
        #out = torch.sigmoid(self.fc1(out))
        #print(out)
        out = self.fc1(out)
       
        return out

class Dncoder_conv3D(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features,13824)
        #self.input_conv1 = nn.Sequential(nn.Conv3d(in_channels=1,out_channels=16,kernel_size=2, stride=3),nn.ELU() ) 
        self.input_conv2 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,out_channels=64,kernel_size=4, stride=3),nn.ELU() ) 
        self.input_conv3 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,out_channels=32,kernel_size=3, stride=3),nn.ELU(), nn.BatchNorm3d(32))
        self.input_conv4 = nn.Sequential(nn.ConvTranspose3d(in_channels=32,out_channels=32,kernel_size=3, stride=3),nn.ELU(),nn.BatchNorm3d(32))
        self.input_conv5 = nn.Sequential(nn.ConvTranspose3d(in_channels=32,out_channels=1,kernel_size=2, stride=3))
      
      
    def forward(self,x):
     
        print(x.size())   
        out = self.fc1(x)
        print(out.size())
        out = out.view(1,-1, 6, 6, 6)
        print(out.size())
        out = self.input_conv2(out) 
        print(out.size())
        out = self.input_conv3(out)
        print(out.size())
        out = self.input_conv4(out)
        print(out.size())
        out = self.input_conv5(out)
        print(out.size())
        out = torch.sigmoid(out)
#         out[out >= 0.5] = 1
#         out[out < 0.5] = 0
        #print(out)
        return out



class AE_3d_conv(nn.Module):
    def __init__(self,hidden_size=4096):
        super().__init__()
        
        self.encoder = Encoder_conv3D(hidden_size)
        self.decoder = Dncoder_conv3D(hidden_size)
        
    def forward(self,x):
        encoder_ = self.encoder(x)
        decoder_ = self.decoder(encoder_)
        return encoder_, decoder_
    

        
    
class Encoder_conv3D_2(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.input_conv2 = nn.Sequential(nn.Conv3d(in_channels=1,out_channels=64,kernel_size=2, stride=3),nn.ELU(),nn.BatchNorm3d(64) ) 
        self.input_conv3 = nn.Sequential(nn.Conv3d(in_channels=64,out_channels=128,kernel_size=2, stride=3),nn.ELU(), nn.BatchNorm3d(128))
        self.input_conv4 = nn.Sequential(nn.Conv3d(in_channels=128,out_channels=256,kernel_size=2, stride=3),nn.ELU(),nn.BatchNorm3d(256))
        self.input_conv5 = nn.Sequential(nn.Conv3d(in_channels=256,out_channels=512,kernel_size=2, stride=3),nn.ELU(),nn.BatchNorm3d(512))
       
        self.fc1 = nn.Linear(110592,1024) # 16**3
        self.fc2 = nn.Linear(1024,out_features)
      
    def forward(self,x):
        #out = self.input_conv1(x)
        print(x.size())
        out = self.input_conv2(x) 
        
        print(out.size())
        out = self.input_conv3(out)
        print(out.size())
        out = self.input_conv4(out)
        print(out.size())
        out = self.input_conv5(out)
        print(out.size())
        out = out.view(out.size(0),-1)
        print(out.size())
        
        #out = torch.sigmoid(self.fc1(out))
        #print(out)
        out = F.relu(self.fc1(out))
        print(out.size())
        out = F.relu(self.fc2(out))
       
        return out

class Dncoder_conv3D_2(nn.Module):
    def __init__(self,in_features):
        super().__init__()
     
        self.fc1 = nn.Linear(in_features,1024)
        self.fc2 = nn.Linear(1024,110592)
        self.input_conv2 = nn.Sequential(nn.ConvTranspose3d(in_channels=512,out_channels=256,kernel_size=4, stride=3),nn.ELU(),nn.BatchNorm3d(256) ) 
        self.input_conv3 = nn.Sequential(nn.ConvTranspose3d(in_channels=256,out_channels=128,kernel_size=3, stride=3),nn.ELU(), nn.BatchNorm3d(128))
        self.input_conv4 = nn.Sequential(nn.ConvTranspose3d(in_channels=128,out_channels=64,kernel_size=3, stride=3),nn.ELU(),nn.BatchNorm3d(64))
        self.input_conv5 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,out_channels=1,kernel_size=2, stride=3))
      
      
    def forward(self,x):
     
        print(x.size())   
        out = F.relu(self.fc1(x))
        print(out.size())
        out = F.relu(self.fc2(out))
        print(out.size())        
        out = out.view(-1,512, 6, 6, 6)
        print(out.size())
        out = self.input_conv2(out) 
        print(out.size())
        out = self.input_conv3(out)
        print(out.size())
        out = self.input_conv4(out)
        print(out.size())
        out = self.input_conv5(out)
        print(out.size())
        out = torch.sigmoid(out)

        return out

class AE_3d_conv_2(nn.Module):
    def __init__(self,hidden_size=4096):
        super().__init__()
        
        self.encoder = Encoder_conv3D_2(hidden_size)
        self.decoder = Dncoder_conv3D_2(hidden_size)
        
    def forward(self,x):
        encoder_ = self.encoder(x)
        decoder_ = self.decoder(encoder_)
        return encoder_, decoder_
    

    
    
    
class Encoder_conv3D_3(nn.Module):
    def __init__(self, out_features):
        super().__init__()

        self.input_conv2 = nn.Sequential(nn.Conv3d(in_channels=1,out_channels=32,kernel_size=3, stride=2),nn.ELU(),nn.BatchNorm3d(32) ) 
        
        self.input_conv3 = nn.Sequential(nn.Conv3d(in_channels=32,out_channels=64,kernel_size=3, stride=3),nn.ELU(), nn.BatchNorm3d(64))
        self.input_conv3_2 = nn.Sequential(nn.Conv3d(in_channels=64,out_channels=256,kernel_size=3, stride=3),nn.ELU(), nn.BatchNorm3d(256))
       
        self.input_conv4 = nn.Sequential(nn.Conv3d(in_channels=256,out_channels=512,kernel_size=2, stride=2),nn.ELU(),nn.BatchNorm3d(512))    
        self.input_conv4_2 = nn.Sequential(nn.Conv3d(in_channels=512,out_channels=512,kernel_size=2, stride=2),nn.ELU(),nn.BatchNorm3d(512))
        
        self.input_conv5 = nn.Sequential(nn.Conv3d(in_channels=512,out_channels=512,kernel_size=2, stride=2),nn.ELU(),nn.BatchNorm3d(512))

        #self.input_conv5_2 = nn.Sequential(nn.Conv3d(in_channels=512,out_channels=512,kernel_size=2, stride=2),nn.ELU(),nn.BatchNorm3d(512))
       
        self.fc1 = nn.Linear(13824,1024) # 16**3
        self.fc2 = nn.Linear(1024,out_features)
      
    def forward(self,x):
        #out = self.input_conv1(x)
        print(x.size())
        out = self.input_conv2(x) 
        print(out.size())
        
        out = self.input_conv3(out)
        print(out.size())
        out = self.input_conv3_2(out)
        print(out.size())
        
        out = self.input_conv4(out)
        print(out.size())
        out = self.input_conv4_2(out)
        print(out.size())
        
        out = self.input_conv5(out)
        print(out.size())
#         out = self.input_conv5_2(out)
#         print(out.size())
        
        out = out.view(out.size(0),-1)
        print(out.size())
        out = F.relu(self.fc1(out))
        print(out.size())
        out = F.relu(self.fc2(out))
       
        return out

class Dncoder_conv3D_3(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.indices1 = None 
        self.indices2 = None 
        
        self.fc1 = nn.Linear(in_features,1024)
        self.fc2 = nn.Linear(1024,13824)
        self.input_conv2 = nn.Sequential(nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=3, stride=2),nn.ELU(),nn.BatchNorm3d(512) ) 
        
        self.input_conv3 = nn.Sequential(nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=2, stride=2),nn.ELU(), nn.BatchNorm3d(512))
        
        self.input_conv4 = nn.Sequential(nn.ConvTranspose3d(in_channels=512,out_channels=256,kernel_size=2, stride=2),nn.ELU(),nn.BatchNorm3d(256))
        
        #self.Unpool3d_2 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.input_conv5 = nn.Sequential(nn.ConvTranspose3d(in_channels=256,out_channels=64,kernel_size=4, stride=3),nn.BatchNorm3d(64))
        
        self.input_conv6 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,out_channels=32,kernel_size=3, stride=3),nn.BatchNorm3d(32))
        
        #self.Unpool3d_1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.input_conv7 = nn.Sequential(nn.ConvTranspose3d(in_channels=32,out_channels=1,kernel_size=4, stride=2),nn.BatchNorm3d(1))
      
      
    def forward(self,x):
     
        print(x.size())   
        out = F.relu(self.fc1(x))
        print(out.size())
        out = F.relu(self.fc2(out))
        print(out.size())        
        out = out.view(-1,512, 3, 3, 3)
        print(out.size())
        out = self.input_conv2(out) 
        print(out.size())
        out = self.input_conv3(out)
        print(out.size())
        out = self.input_conv4(out)
        print(out.size())
        
        out = self.input_conv5(out)
        print(out.size())
        out = self.input_conv6(out)
        print(out.size())
        
        out = self.input_conv7(out)
        print(out.size())
        out = torch.sigmoid(out)
        return out

class AE_3d_conv_3(nn.Module):
    def __init__(self,hidden_size=4096):
        super().__init__()
        
        self.encoder = Encoder_conv3D_3(hidden_size)
        self.decoder = Dncoder_conv3D_3(hidden_size)
      
    def forward(self,x):
        encoder_ = self.encoder(x)
        decoder_ = self.decoder(encoder_,)
        return encoder_, decoder_
    
    
    
    
class Encoder_conv3D_4(nn.Module):
    def __init__(self, out_features):
        super().__init__()

        self.input_conv2 = nn.Sequential(nn.Conv3d(in_channels=1,out_channels=32,kernel_size=3, stride=2),nn.ELU(),nn.BatchNorm3d(32) ) 
        
        self.input_conv3 = nn.Sequential(nn.Conv3d(in_channels=32,out_channels=64,kernel_size=3, stride=2),nn.ELU(), nn.BatchNorm3d(64))
        self.input_conv3_2 = nn.Sequential(nn.Conv3d(in_channels=64,out_channels=256,kernel_size=3, stride=3),nn.ELU(), nn.BatchNorm3d(256))
       
        self.input_conv4 = nn.Sequential(nn.Conv3d(in_channels=256,out_channels=512,kernel_size=2, stride=2),nn.ELU(),nn.BatchNorm3d(512))    
        self.input_conv4_2 = nn.Sequential(nn.Conv3d(in_channels=512,out_channels=512,kernel_size=2, stride=2),nn.ELU(),nn.BatchNorm3d(512))
        
        self.input_conv5 = nn.Sequential(nn.Conv3d(in_channels=512,out_channels=512,kernel_size=2, stride=2),nn.ELU(),nn.BatchNorm3d(512))

        #self.input_conv5_2 = nn.Sequential(nn.Conv3d(in_channels=512,out_channels=512,kernel_size=2, stride=2),nn.ELU(),nn.BatchNorm3d(512))
       
        self.fc1 = nn.Linear(13824,4096) # 16**3
        self.fc2 = nn.Linear(4096,out_features)
      
    def forward(self,x):
        #out = self.input_conv1(x)
        print(x.size())
        out = self.input_conv2(x) 
        print(out.size())
        
        out = self.input_conv3(out)
        print(out.size())
        out = self.input_conv3_2(out)
        print(out.size())
        
        out = self.input_conv4(out)
        print(out.size())
        out = self.input_conv4_2(out)
        print(out.size())
        
        out = self.input_conv5(out)
        print(out.size())
        
        out = out.view(out.size(0),-1)
        print(out.size())
        out = F.relu(self.fc1(out))
        print(out.size())
        out = F.relu(self.fc2(out))
       
        return out

class Dncoder_conv3D_4(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.indices1 = None 
        self.indices2 = None 
        
        self.fc1 = nn.Linear(in_features,4096)
        self.fc2 = nn.Linear(4096,13824)
        self.input_conv2 = nn.Sequential(nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=2, stride=2),nn.ELU(),nn.BatchNorm3d(512) ) 
        
        self.input_conv3 = nn.Sequential(nn.ConvTranspose3d(in_channels=512,out_channels=512,kernel_size=2, stride=2),nn.ELU(), nn.BatchNorm3d(512))
        
        self.input_conv4 = nn.Sequential(nn.ConvTranspose3d(in_channels=512,out_channels=256,kernel_size=3, stride=2),nn.ELU(),nn.BatchNorm3d(256))
        
        #self.Unpool3d_2 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.input_conv5 = nn.Sequential(nn.ConvTranspose3d(in_channels=256,out_channels=64,kernel_size=4, stride=3),nn.BatchNorm3d(64))
        
        self.input_conv6 = nn.Sequential(nn.ConvTranspose3d(in_channels=64,out_channels=16,kernel_size=3, stride=2),nn.BatchNorm3d(16))
        
        #self.Unpool3d_1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.input_conv7 = nn.Sequential(nn.ConvTranspose3d(in_channels=16,out_channels=1,kernel_size=4, stride=2),nn.BatchNorm3d(1))
      
      
    def forward(self,x):
     
        print(x.size())   
        out = F.relu(self.fc1(x))
        print(out.size())
        out = F.relu(self.fc2(out))
        print(out.size())        
        out = out.view(-1,512, 3, 3, 3)
        print(out.size())
        out = self.input_conv2(out) 
        print(out.size())
        out = self.input_conv3(out)
        print(out.size())
        out = self.input_conv4(out)
        print(out.size())
        
        out = self.input_conv5(out)
        print(out.size())
        out = self.input_conv6(out)
        print(out.size())
        
        out = self.input_conv7(out)
        print(out.size())
        #out = torch.sigmoid(out)
        return out

class AE_3d_conv_scaled(nn.Module):
    def __init__(self,hidden_size=4096):
        super().__init__()
        
        self.encoder = Encoder_conv3D_4(hidden_size)
        self.decoder = Dncoder_conv3D_4(hidden_size)
      
    def forward(self,x):
        encoder_ = self.encoder(x)
        decoder_ = self.decoder(encoder_,)
        return encoder_, decoder_