import torch
import torch.nn as nn
import numpy as np
from data_loader import get_loader,CelebA
from torchvision.models import inception_v3
import logger
import time,datetime
import random
import torch.nn.functional as F
import sys
from torchvision import transforms as T

def classification_loss(logit, target, dataset='CelebA'):
    """Compute binary or softmax cross entropy loss."""
    if dataset == 'CelebA':
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
    elif dataset == 'RaFD':
        return F.cross_entropy(logit, target)

def train_inc(config,device,inc_net):
    #Inception network is trained to classify all attributes!!!
    image_size=299 #According to inception network
    lr=0.0001
    log_step=1
    opt=torch.optim.Adam(inc_net.parameters(),lr,[0.5,0.999])

    train_dataset=get_loader(config.celeba_image_dir, config.attr_path, 
        config.selected_attrs,image_size=image_size,num_workers=config.num_workers,
        dataset=config.dataset)

    test_dataset=get_loader(config.celeba_image_dir, config.attr_path, 
    config.selected_attrs,image_size=image_size,num_workers=config.num_workers,
    dataset=config.dataset,mode='test')

    print('Start training...')
    start_time=time.time()
    max_acc=0
    for p in range(100):
        for i,data in enumerate(train_dataset):
            img, label = data
            img.to(device)
            label.to(device)
            batch_pred = inc_net(img)
            loss=classification_loss(batch_pred,label,config.dataset)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if i%log_step==0:
                et=time.time()-start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}] , loss [{}]".format(et, i+1,p,loss.item())
                print(log)

                acc=0
                with torch.no_grad():
                    for i,data in enumerate(test_dataset):
                        img_test,label_test=data
                        label_test=label_test[:,:len(config.selected_attrs)]
                        pred=inc_net(img_test)
                        pred_label=pred>0.5
                        #or test_label=torch.round(pred)
                        acc+=torch.mean(torch.eq(label_test,pred_label.type(torch.FloatTensor)).type(torch.FloatTensor))
                acc/=len(test_dataset)
                print("Test Accuracy: ", acc.data[0])
                if acc>max_acc:
                    torch.save(inc_net.state_dict(),path)
                    max_acc=acc


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

def score(config,Gen, train=False):
    
    #Inception net
    inc_net=inception_v3(pretrained=False, num_classes=len(config.selected_attrs),aux_logits=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inc_net.to(device)
    
    if train:
        print("Training Inception network...")
        train_inc(config,device,inc_net)
    
    #Get dataset
    test_dataset=get_loader(config.celeba_image_dir, config.attr_path, 
        config.selected_attrs,image_size=config.image_size,num_workers=config.num_workers,
        dataset=config.dataset,batch_size=config.batch_size,mode='test')
    
    transform=[]
    transform.append(T.ToPILImage())
    transform.append(T.Resize(299))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))
    transform=T.Compose(transform)
    
    data_iter=iter(test_dataset)
    len_sel_labels=len(config.selected_attrs)

    if config.dataset=='CelebA':
        hair_color_indices=[]
        for i,attr_name in enumerate(config.selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)
    Gen.to(device)

    mean_,steps=0,2
    print("Calculating score...")
    with torch.no_grad():
        for i in range(steps):
            try:
                img, all_labels=next(data_iter)
            except:
                data_iter=iter(data_iter)
                img,all_labels=next(data_iter) #label is a boolean labelled vector
            
            #randomly flip  
            print("Label flipping")
            flipped_labels=flip_labels(all_labels,config.selected_attrs,config.dataset,hair_color_indices)
            print("Labels flipped")
            
            img=img.to(device)
            flipped_labels=flipped_labels.to(device)
            #Obtain probabilities of Generated samples!

            x_gen=Gen(img,flipped_labels)
            
            x_gen=torch.stack([transform(pop.detach().cpu()) for pop in x_gen])
            print(x_gen.size())

            x_gen=x_gen.to(device)
            
            pred_x_gen=inc_net(x_gen)
            bCE=flipped_labels*torch.log(pred_x_gen)+(1-flipped_labels)*torch.log(1-pred_x_gen)
            mean_+=torch.mean(torch.sum(bCE,1)) #Can be mean!!!???

    return mean_/steps
                    


