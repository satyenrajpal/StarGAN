import torch
import torch.nn as nn
import numpy as np
from data_loader import get_loader,CelebA
from torchvision.models import inception_v3
import logger
import time,datetime
import random
import torch.nn.functional as F
import sys,os
from torchvision import transforms as T


class InceptionNet():
    def __init__(self,config):
        self.image_size=299 #Inception net condition
        self.lr=0.0001
        self.log_step=100
        self.selected_attrs=config.selected_attrs
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buildIncNet()
        
        self.save_incDir=config.inc_net_dir
        self.pretrained_incNet=config.pretrained_incNet
        self.dataset=config.dataset
        self.test_dataset=get_loader(config.celeba_image_dir, config.attr_path, 
                                    config.selected_attrs,image_size=self.image_size,num_workers=config.num_workers,
                                    dataset=config.dataset,mode='test')
    
    def buildIncNet(self):
        self.inc_net=inception_v3(pretrained=False, num_classes=len(self.selected_attrs),aux_logits=False)
        self.opt=torch.optim.Adam(self.inc_net.parameters(),self.lr,[0.5,0.999])
        self.inc_net.to(self.device)

    def load_pretrained(self):
        if self.pretrained_incNet is not None:
            self.inc_net.load_state_dict(torch.load(self.pretrained_incNet,map_location=lambda storage, loc:storage))
        else:
            sys.exit("Pretrained path invalid")

    @staticmethod
    def classification_loss(logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self,config):
        train_dataset=get_loader(config.celeba_image_dir, config.attr_path, 
                                config.selected_attrs,image_size=self.image_size,
                                num_workers=config.num_workers,dataset=config.dataset)
        
        print('Start Training...')
        start_time=time.time()
        max_acc,epochs=0,50
        for p in range(epochs):
            for i,data in enumerate(train_dataset):
                img, label = data
                img=img.to(self.device)
                label=label.to(self.device)
                
                batch_pred = self.inc_net(img)
                loss=self.classification_loss(batch_pred,label,config.dataset)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                if i%self.log_step==0:
                    et=time.time()-start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    
                    acc=self.test()
                    print("Test Accuracy: ", acc)
                    if acc>max_acc:
                        path=os.path.join(self.save_incDir,'{}-{}-incNet.ckpt'.format(p,i))
                        torch.save(self.inc_net.state_dict(),path)
                        max_acc=acc
                    
                    log = "Elapsed [{}], Epoch[{}] - Iteration [{}/{}] , loss [{}], max_acc[{}]".format(et, p,i+1,len(train_dataset),loss.item(),max_acc)
                    print(log)
                    
    def test(self):
        acc=0
        with torch.no_grad():
            for _,data in enumerate(self.test_dataset):
                img_test,label_test=data
                label_test=label_test[:,:len(self.selected_attrs)]
                
                img_test=img_test.to(self.device)
                label_test=label_test.to(self.device)
                
                pred=self.inc_net(img_test)
                pred_label=pred>0.5
                pred_label=pred_label.type(torch.FloatTensor).to(self.device)
                acc+=torch.mean(torch.eq(label_test,pred_label).type(torch.FloatTensor).to(self.device)).item()
        
        acc/=len(self.test_dataset)
        return acc

    @staticmethod
    def flip_labels(labels,selected_attrs,dataset,hair_color_indices=None):
        """ Flip trained labels randomly 
        Inputs:
            labels: labels corresponding to image (selected_labels+n_selectedLabels)
            selected_attrs: selected attributes [List]
            dataset: 'CelebA' or 'RaFD'
        Return:
            flipped labels that the model was trained on 
                Shape - [batch_size,len(selected_attrs)] 
        """
        flipped=labels.clone()
        flipped=flipped[:,:len(selected_attrs)] #discard labels that were not trained on
        if dataset=='CelebA':
            for i in range(len(flipped)):
                if hair_color_indices is not None:
                    h=torch.zeros(len(hair_color_indices))
                    h[random.randint(0,len(hair_color_indices)-1)] =1
                count=0
                for j in range(len(selected_attrs)):
                    if hair_color_indices is not None and j in hair_color_indices:
                        flipped[i,j]=h[count]
                        count+=1
                    else:
                        flipped[i,j]=random.randint(0,1)
        return flipped

    def transform_op(self,image_size):
        transform=[]
        transform.append(T.ToPILImage())
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
        return T.Compose(transform)

    def score(self,Gen):
        
        #Inception net
        self.load_pretrained()

        if config.dataset=='CelebA':
            hair_color_indices=[]
            for i,attr_name in enumerate(self.selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)
        
        mean_,steps=0,2
        transform=self.transform_op(self.image_size)
        
        Gen.to(device)
        sigmoid=nn.Sigmoid()
        data_iter=iter(test_dataset)

        print("Calculating score...")
        with torch.no_grad():
            for i in range(steps):
                try:
                    img, all_labels=next(data_iter)
                except:
                    data_iter=iter(data_iter)
                    img,all_labels=next(data_iter) #label is a boolean labelled vector
                
                #randomly flip  
                flipped_labels=self.flip_labels(all_labels,self.selected_attrs,self.dataset,hair_color_indices)
                
                img=img.to(device)
                flipped_labels=flipped_labels.to(device)

                x_gen=Gen(img,flipped_labels)
                x_gen=torch.stack([transform(pop.detach().cpu()) for pop in x_gen])
                x_gen=x_gen.to(device)

                pred_x_gen=sigmoid(self.inc_net(x_gen))
                bCE=flipped_labels*torch.log(pred_x_gen)+(1-flipped_labels)*torch.log(1-pred_x_gen)
                mean_+=torch.mean(torch.sum(bCE,1)).cpu().item() 
                print(mean_)
        
        return mean_/steps