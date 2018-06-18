from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np
import sys
import h5py
class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split() #<- attribute names as a list
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1') #each attribute converted to 1 hot vector

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        if self.transform is not None:
            return self.transform(image), torch.FloatTensor(label)
        else:
            return np.array(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class CelebA_HQ(data.Dataset):
    """ Dataset for CelebA HQ dataset"""

    def __init__(self,h5_path,hq_attr_path,attr_path,selected_attrs,transform,mode,step=0):
        """Initialize and preprocess the CelebA dataset."""
        self.h5_path = h5_path
        self.hq_attr_path=hq_attr_path
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.HDF5_dataset = self.preprocess(step)

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self,step):
        """Preprocess the CelebA-HQ attribute file with regards to the CelebA attribute file."""
        #Process h5 file
        h5_dataset=h5py.File(self.h5_path,'r')
        lods=sorted([value for key,value in h5_dataset.items() if key.startswith('data')],key=lambda lod: -lod.shape[3])
        lods=lods[:6] #Require datasets form 1024^2 to 32^2 resolution

        # Store only the required attributes from the celebA attribute file 
        cA_lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = cA_lines[1].split() #<- attribute names as a list
        # Attributes -> Indices and the other way around
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        cA_lines = cA_lines[2:] #[Filename attr#1 attr#2 ....,Filename attr#1 attr#2 ....,...] 
        
        #Obtain attributes in accordance to celebA-HQ dataset
        cAHQ_lines=[line.rstrip() for line in open(self.hq_attr_path,'r')]
        cAHQ_lines=cAHQ_lines[1:] #<- Remove headings from the file
        for i,line in enumerate(cAHQ_lines):
            split=line.split()
            filename=split[2]
            idx=int(split[0]) #<Index corresponding to image in HDF5 dataset
            attr_line=cA_lines[int(split[1])] #<- Corresponding idx in celebA file
            attr_values=attr_line.split()[1:] #<- [0] is filename

            label=[]
            for attr_name in self.selected_attrs:
                idx=self.attr2idx[attr_name]
                label.append(attr_values[idx]=='1')

            if (i+1) < 1000:
                self.test_dataset.append([idx,filename,label])
            else:
                self.train_dataset.append([idx,filename,label])

        print('Finished preprocessing the CelebA-HQ dataset...')
        return lods[len(lods)-step-1]

    def __getitem__(self,index):
        dataset=self.train_dataset if mode=='train' else self.test_dataset
        idx,filename, label = dataset[index]
        image=self.HDF5_dataset[idx]
        image=image.transpose(1,2,0) # CHW => HWC
        if self.transform is not None:
            return self.transform(image), torch.FloatTensor(label)
        else:
            return np.array(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_images

def get_loader(dict_,step=0,batch_size=16):
    """Build and return a data loader."""
    transform = []
    if dict_['dataset'] == 'CelebA-HQ':
        transform.append(T.ToPILImage()) #Convert to PIL to perform operations
    if dict_['mode'] == 'train':
        transform.append(T.RandomHorizontalFlip())
    if dict_['dataset'] == 'CelebA' or dict_['dataset'] == 'RaFD':
        transform.append(T.CenterCrop(dict_['crop_size']))
        transform.append(T.Resize(int(2**(5+step)))) #32 -> 64 -> 128...
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    
    if dict_['dataset'] == 'CelebA':
        dataset = CelebA(dict_['img_dir'], dict_['attr_path'], dict_['selected_attrs'], transform, dict_['mode'])
    elif dict_['dataset'] == 'RaFD':
        dataset = ImageFolder(dict_['img_dir'], transform)
    elif dict_['dataset'] == 'CelebA-HQ':
        dataset = CelebA_HQ(dict_['h5_path'],dict_['hq_attr_path'],
            dict_['attr_path'],dict_['selected_attrs'],transform,dict_['mode'],step)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(dict_['mode']=='train'),
                                  num_workers=dict_['num_workers'])
    return data_loader