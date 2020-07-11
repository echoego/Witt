#dataset
import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, frames,labels=None,transform=None,load_all=False):
        "Initialization"
        self.data_path = data_path
        #self.labels = labels
        folders=[]
        for f in os.listdir(data_path):
            if f!='.DS_Store':
                folders.extend(list(map(lambda x:f+'/'+x,os.listdir(data_path+f))))
                
        for i in range(len(folders)-1,-1,-1):
            if '.DS_Store' in folders[i]:
                folders.pop(i)
        
        self.folders = folders
        self.load_all=load_all
        self.transform = transform
        self.frames = frames
        
        if self.load_all:
            k=0
            temp=[]
            print('loading all images')
            for f in self.folders:
                k+=1
                temp.append(self.read_images(self.data_path, f, self.transform) )
                if (k)%1000==0:
                    print(float(k)/len(self.folders))
            self.dataset=temp
        
        

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame'+str(i)+'.jpg'))

            if use_transform is not None:
                image1 = use_transform(image)

            X.append(image1)
            image.close()
        X = torch.stack(X, dim=0)

        return X


    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        if not self.load_all:
            folder = self.folders[index]

        # Load data
            X = self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        #y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor
        else:
            X=self.dataset[index]
        return X

    

