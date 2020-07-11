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

        
        
class DecoderNet(nn.Module):
    def __init__(self,in_features):
        super(DecoderNet, self).__init__()
        self.decoder_fl=nn.Linear(in_features=in_features,out_features=56*75*3)
        self.decoder_f2=nn.ConvTranspose2d(in_channels=3,out_channels=64,kernel_size=(7,7),stride=2)
        self.decoder_f3=nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=(8,12),stride=2)

    def forward(self,x):
        temp=self.decoder_fl(x)
        temp=temp.view(-1,3,56,75)
        temp=self.decoder_f2(temp)
        x_rec=self.decoder_f3(temp)
        return(x_rec)



class VaeNet(nn.Module):
    def __init__(self,device,img_x=240,img_y=320,z_dim=300):
        super(VaeNet, self).__init__()
        self.img_x=img_x
        self.img_y=img_y
        self.z_dim=z_dim
        self.encoder=EncoderCNN(img_x=self.img_x, img_y=self.img_y, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=z_dim)
        self.decoder=DecoderNet(in_features=self.z_dim)
        self.layer_mu=nn.Linear(in_features=self.z_dim,out_features=self.z_dim)
        self.layer_logvar=nn.Linear(in_features=self.z_dim,out_features=self.z_dim)
        self.device=device

    
    def reparamentize(self,mu,logvar):
        z=mu + Variable(torch.randn(mu.size(0),mu.size(1)).to(self.device))*(logvar/2).exp()
        return(z)

    def decode(self,z_temp):
        mu=self.layer_mu(z_temp)
        logvar=self.layer_logvar(z_temp)
        z=self.reparamentize(mu,logvar)
        rec_x=self.decoder(z)
        return(rec_x,mu,logvar)

    def forward(self,x):
        z_temp=self.encoder(x)
        rec_x,mu,logvar=self.decode(z_temp)
        return(rec_x,mu,logvar)


def Loss(x,rec_x,mean,logvar):
    bce_loss=F.binary_cross_entropy(input=rec_x,target=x,size_average=False)
    bld_loss=0.5*torch.sum(mean.pow(2)+logvar.exp()-logvar-1)
    return(bce_loss+bld_loss)


if __name__=='__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x=torch.tensor(np.random.rand(3*240*320).reshape(1,3,240,320),dtype=torch.float32)
    vaenet=VaeNet(device=device)
    rec_x,mu,logvar=vaenet(x)



