import torch
import torch.nn as nn
import numpy as np
from data_loader import get_loader
from torchvision.model import inception_v3
import logger
import time,datetime


def classification_loss(logit, target, dataset='CelebA'):
    """Compute binary or softmax cross entropy loss."""
    if dataset == 'CelebA':
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
    elif dataset == 'RaFD':
        return F.cross_entropy(logit, target)

def train_inc(config,device,inc_net):
    image_size=299 #According to inception network
    lr=0.0001
    log_step=100
    opt=torch.optim.Adam(inc_net.parameters(),lr,[0.5,0.999])

    train_dataset=get_loader(config.image_dir, config.attr_path, 
        config.selected_attrs,image_size=image_size,num_workers=config.num_workers,
        dataset=config.dataset)

    test_dataset=get_loader(config.image_dir, config.attr_path, 
    config.selected_attrs,image_size=image_size,num_workers=config.num_workers,
    dataset=config.dataset,mode='test')

    print('Start training...')
    start_time=time.time()
    for p in range(100):
        for i,data in enumerate(train_dataset):
            img, label = data
            img.to(device)
            label.to(device)
            batch_pred = inc_net(data)
            loss=classification_loss(batch_pred,label,dataset)
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
                        pred=inc_net(img_test)
                        pred_label=pred>0.5
                        #or test_label=torch.round(pred)
                        acc+=torch.mean(torch.eq(label_test,pred_label))
                acc/=len(test_dataset)
                print("Test Accuracy: ", acc)



def inception_score(config,device,train=False):
    
    if train:
        train_inc(config,device)
    test_dataset=get_loader(config.image_dir, config.attr_path, 
        config.selected_attrs,image_size=image_size,num_workers=config.num_workers,
        dataset=config.dataset,mode='test')

    inc_net=inception_v3(pretrained=False, classes=len(config.selected_attrs),aux_logits=False)
    inc_net.to(device)

    for i, data in enumerate(test_dataset):
        img,label=data
        pred=

                    


