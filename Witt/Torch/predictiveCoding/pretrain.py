from InertiaNet import *
from data import *
import torch
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle


d_path='/Users/lekang/anaconda/tests/Review/Torch/predictiveCoding/ucf101-jpg/'
action_name_path = "inertialCodingNet.pkl"
save_model_path = "checkpoints/"

transform = transforms.Compose([transforms.Resize([240, 320]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


datatest=Dataset_CRNN(data_path=d_path,
                      frames=list(range(1,21)),
                      transform=transform,
                      load_all=False
                     )

use_cuda=False
batch_size=20
lr=1e-3
epoch=10

all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 6}#, 'pin_memory': True}# if use_cuda else {}
all_data_loader = data.DataLoader(datatest,**all_data_params)


code_net=inertiaCodingNet()

for ep in range(epoch):
    for i,datai in eumerrat


                

