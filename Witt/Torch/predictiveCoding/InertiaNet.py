import torch
import numpy as np
from torch.functional import F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torchvision.datasets as dst
from torchvision.utils import save_image
import pandas as pd

img_x=240
img_y=320

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

#二维卷积
class EncoderCNN(nn.Module):
    def __init__(self, img_x=img_x, img_y=img_y, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1)   # fully connected layer, output k classes
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)   # output = CNN embedding latent variables

    def forward(self, x):
        
            # CNNs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv

        # FC layers
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        out = self.fc3(x)
            
        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return out

#class: sequence coding module
#惰性编码单元
class inertiaCodingCell(nn.Module):
    '''
    input:(batch,seq_len,channel,witdth,height)
    output:(batch,code_len,code_dim)
    
    '''
    def __init__(self,n_channel=1,width=img_x,height=img_y):
        super(inertiaCodingCell,self).__init__()
        self.encoder_l=EncoderCNN(CNN_embed_dim=1024)
        self.d_f1=nn.Linear(in_features=1024,out_features=1024)
        self.d_fE=nn.Linear(in_features=1024,out_features=256)
        self.d_fS=nn.Linear(in_features=1024,out_features=256)
        
        self.pred_f1=nn.Linear(in_features=512,out_features=512)
        self.pred_f2=nn.Linear(in_features=512,out_features=256)
        
        self.decoder_fl=nn.Linear(in_features=512,out_features=56*75*3)
        self.decoder_f2=nn.ConvTranspose2d(in_channels=3,out_channels=64,kernel_size=(7,7),stride=2)
        self.decoder_f3=nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=(8,12),stride=2)

                
    def encoder(self,x):
        h=self.encoder_l(x)
        h=self.d_f1(h)
        h=F.relu(h)
        E=self.d_fE(h)
        E=F.relu(E)
        S=self.d_fS(h)
        S=F.relu(S)
        return E,S
    
    def predictor(self,E,S):
        temp=torch.cat([E,S],1)
        temp=self.pred_f1(temp)
        temp=F.relu(temp)
        out=self.pred_f2(temp)
        out=F.relu(out)
        return(out)

    def decoder(self,E,S):
        temp=torch.cat([E,S],1)
        temp=self.decoder_fl(temp)
        temp=temp.view(-1,3,56,75)
        temp=self.decoder_f2(temp)
        x_rec=self.decoder_f3(temp)
        return(x_rec)
        
    def forward(self,x,pre_E=None,pred_S=None):
        
        if (pre_E is None) or (pred_S is None):
            
            x_in=x
            E_plus,S_plus=self.encoder(x_in)
            E=E_plus
            S=S_plus
            x_rec=self.decoder(E,S)
            S_pred=self.predictor(E,S)
            
        else:
            
            x_pred_rec=self.decoder(pre_E,pred_S)
            x_in=x-x_pred_rec
            E_plus,S_plus=self.encoder(x_in)
            E=pre_E+E_plus
            S=pred_S+S_plus
            x_rec=self.decoder(E,S)
            S_pred=self.predictor(E,S)
        
        return(E,S,S_pred,E_plus,x_rec)
    
#惰性编码预测网络
class inertiaCodingNet(nn.Module):
    def __init__(self,seq_len=10,n_channel=3,w=img_x,h=img_y):
        super(inertiaCodingNet,self).__init__()
        self.seq_len=seq_len
        self.n_channel=n_channel
        self.w=w
        self.h=h
        self.inertiaCell=inertiaCodingCell()
        
        
    
    #input (batch_size,seq_len,n_channel,w,h)
    def forward(self,x):
        b,s_len,c_len,w,h=x.shape
        E=[]
        S=[]
        S_pred=[]
        E_plus=[]
        x_rec=[]
        for i in range(s_len):
            x_t=x[:,i,:,:,:]
            
            if i==0:
                E_t,S_t,S_pred_t,E_plus_t,x_rec_t=self.inertiaCell(x_t)
            else:
                E_t,S_t,S_pred_t,E_plus_t,x_rec_t=self.inertiaCell(x_t,pre_E=E_t,pred_S=S_pred_t)
            
            E.append(E_t)
            S.append(S_t)
            S_pred.append(S_pred_t)
            E_plus.append(E_plus_t)
            x_rec.append(x_rec_t)
            
        E=torch.stack(E).transpose(0,1)
        S=torch.stack(S).transpose(0,1)
        E_plus=torch.stack(E_plus).transpose(0,1)
        x_rec=torch.stack(x_rec).transpose(0,1)
            
        #(batch_size,seq_len,hidden_dim)
        return(E,S,E_plus,x_rec)
        
        
#方差损失

def var_loss(tensor):
    temp=tensor
    means=torch.mean(temp,axis=1).view(tensor.shape[0],1,-1)
    return torch.sum(torch.pow(temp-means,2))
        
        
        
        
        
