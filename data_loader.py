from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os, sys
import random
import shutil, glob

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
        self.discard_labels=['5_o_Clock_Shadow','Blurry','Double_Chin','Narrow_Eyes','No_Beard','Oval_Face','Wavy_Hair','Bangs']
        self.preprocess()
        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        # random.shuffle(lines)
        shuffle = False
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            nSel_labels,sel_labels = [],[]
            for attr_name in all_attr_names:
                idx = self.attr2idx[attr_name]
                if attr_name in self.selected_attrs:
                    sel_labels.append(values[idx] == '1')
                elif attr_name not in self.discard_labels:
                    nSel_labels.append(values[idx]== '1')
            all_labels = sel_labels + nSel_labels

            if (i+1) < 2000:
                self.test_dataset.append([filename, all_labels])
            else:
                if not shuffle:
                    random.shuffle(lines[2000:])
                    shuffle = True
                self.train_dataset.append([filename, sel_labels])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

def pre_RaFD(root):
    """ Segregate RafD images into folders """
    cwd = os.getcwd()
    os.chdir(root)
    attr = ['angry','contemptuous','disgusted','fearful','happy','neutral','sad', 'surprised']
    print("Folders created. Segregating images into folders..")
    for name in attr:
        if not os.path.exists(name):
            os.makedirs(name)
    for i, file in enumerate(glob.glob("*.jpg")):
        for name in attr:
            if name in file:
                shutil.move(file,os.path.join(name,file))
                break
    os.chdir(cwd)
    print("Preprocessing of RaFD done.")

class RaFD_dataset(data.Dataset):
    def __init__(self, img_dir, transform, mode='train'):
        self.transform = transform
        self.train_dataset = []
        self.test_dataset=[]
        self.mode = mode
        self.img_dir = img_dir
        
        self.preprocess(img_dir)
    
    def preprocess(self, img_dir):
        cwd = os.getcwd()
        os.chdir(img_dir)
        
        attr = ['angry','contemptuous','disgusted','fearful','happy','neutral','sad', 'surprised']
        angles = [0, 45, 90, 135, 180]
        for i, filename in enumerate(glob.glob("*.jpg")):
            split = filename.split("_")
            angle = int(split[0][-3:])
            expression = split[4]
            gender = split[3]
            race = split[2]
            if i<100:
                self.test_dataset.append([filename, attr.index(expression), angles.index(angle)])
            else:
                self.train_dataset.append([filename, attr.index(expression), angles.index(angle)])
        os.chdir(cwd)

    def __getitem__(self, index):
        
        dataset = self.train_dataset if self.mode=='train' else self.test_dataset
        filename, expression_idx, angle_idx = dataset[index]

        image = Image.open(os.path.join(self.img_dir, filename))
        img_transformed = self.transform(image)
        
        return img_transformed, torch.FloatTensor([expression_idx]), torch.FloatTensor([angle_idx])

    def __len__(self):
        return len(self.train_dataset) if self.mode=='train' else len(self.test_dataset) 

def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1, pose=False):
    
    """Build and return a data loader."""
    transform = []
    if mode == 'train' and not pose:
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    print("transform: ", transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD' and not pose:
        dataset = ImageFolder(image_dir, transform)
    elif dataset == 'RaFD' and pose:
        print("Fetching RafD Pose dataset")
        dataset = RaFD_dataset(image_dir, transform)
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader