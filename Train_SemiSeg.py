import os
import numpy as np
import torch
from torch.utils.data import DataLoader,ConcatDataset
import argparse
import random
import math

from networks.StereoNet import Feature_extractor, StereoMatch
from networks.ClassNet import resnet50
from Dataset import SegLabelData, SegUnlabelDatav3, SegValData
from Loss import SegLoss, CEWeightLoss, ContrastiveLoss, EntropyLoss, ConsisLoss
from utils.DataAugmentation import StrongAugment, WeakAugment
from utils.FeatureMemory import PixelFeatureMemory, ObjectFeatureMemory

from networks.DeepLabv3plus.modeling import deeplabv3plus_resnet50
from common.metrics import IOU_mean
from common.ImgProcess import Warp


def adjust_learning_rate(optimizer):
    lr_ft=optimizer.param_groups[0]['lr']
    lr_ft=lr_ft*0.9
    if lr_ft>5e-5:
        optimizer.param_groups[0]['lr']=lr_ft
        
    lr=optimizer.param_groups[1]['lr']
    lr=lr*0.9
    if lr>1e-4:
        optimizer.param_groups[1]['lr']=lr
        
        
def teacher_init(student,teacher):
    for param in teacher.parameters():
        param.detach_()
    student_params = list(student.parameters())
    teacher_params = list(teacher.parameters())
    for i in range(len(student_params)):
        teacher_params[i].data[:] = student_params[i].data[:].clone()

    return teacher


def teacher_update(student,teacher,alpha,i_iter):
    student_params = list(student.parameters())
    teacher_params = list(teacher.parameters())
    alpha= min(1 - 1 / (i_iter*10 + 1), alpha)
    for t_param, s_param in zip(teacher_params, student_params):
        t_param.data[:] = alpha * t_param[:].data[:] + (1 - alpha) * s_param[:].data[:]

    return teacher


def ramp_up(i_iter, max_iter):
    if i_iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - float(i_iter) / float(max_iter)) ** 2)
    
    
def gen_class_data(imgs,probs,num_classes):
    #imgs: [B,3,H,W], probs: [B,14,H,W]
    labels_class=[]
    inp_list=[]
    for i in range(imgs.shape[0]):
        img=imgs[i]  #[3,H,W]
        prob=probs[i]  #[14,H,W]
        for j in range(num_classes):
            mask=prob[j:j+1]  #[1,H,W]
            if torch.max(mask)>=0.9:
                inp_list.append(mask*img)  #[3,H,W]
                labels_class.append(j)
            
    inps=torch.stack(inp_list).cuda()  #[N,3,H,W]
    labels_class=torch.tensor(labels_class).cuda()  #[N]
    return inps,labels_class
        
        
def train(args):
    lr_adjust_epoch=50
    epoch_dict,miou_dict={},{}
    for i in range(1,args.save_num+1):
        epoch_dict[str(i)]=0
        miou_dict[str(i)]=0
    best_miou=0
    best_epoch=0
    epoch=0
    RAMP_UP_ITERS = 1000
    
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root,exist_ok=True)
            
    trainset_label=SegLabelData(args.labeled_img_path)
    trainset_unlabel=SegUnlabelDatav3(args.unlabeled_img_path)
    testset1=SegValData(os.path.join(args.data_root,'Real/val'))
    testset2=SegValData(os.path.join(args.data_root,'Syn/val'))
    testset=ConcatDataset([testset1,testset2])
    num_label=len(trainset_label)
    
    trainloader_label = DataLoader(dataset=trainset_label,batch_size=args.batch_size,shuffle=True,drop_last=True)
    trainloader_unlabel = DataLoader(dataset=trainset_unlabel,batch_size=args.batch_size,shuffle=True,drop_last=True)
    trainloader_label_iter = iter(trainloader_label)
    trainloader_unlabel_iter = iter(trainloader_unlabel)
    
    testloader = DataLoader(dataset=testset,batch_size=args.batch_size,shuffle=False)
    
    num_classes=14
    #Student model
    student=deeplabv3plus_resnet50(num_classes=14, output_stride=16, pretrained_backbone=True)
    student.cuda() 
    student.train()
    print('Model parameters: {}'.format(sum(param.numel() for param in student.parameters())))
        
    #Teacher model
    teacher=deeplabv3plus_resnet50(num_classes=14, output_stride=16, pretrained_backbone=True)
    teacher.cuda()
    teacher=teacher_init(student,teacher)
    teacher.train()
    
    if args.pre_train:
        checkpoint=torch.load(args.model_path)
        student.load_state_dict(checkpoint['student'])
        teacher.load_state_dict(checkpoint['teacher'])
        
    #Load disparity model
    disp_FE=Feature_extractor(in_channel=3,base_channel=16)
    scale_dict={'d2':np.linspace(-1,3,20),'full':np.linspace(-0.2,0.6,20)}
    disp_match=StereoMatch(filter_num=3,scale_dict=scale_dict,group_num=16)
    disp_FE.cuda()
    disp_match.cuda()
    checkpoint=torch.load(args.disp_model_path)
    disp_FE.load_state_dict(checkpoint['FE'])
    disp_match.load_state_dict(checkpoint['Match'])
    disp_FE.eval()
    disp_match.eval()
    
    #Load classification model
    class_model=resnet50(num_classes, include_top=True)
    class_model.cuda()
    class_model=torch.nn.DataParallel(class_model)
    class_model.load_state_dict(torch.load(args.class_model_path))
    class_model.eval()
    for param in class_model.parameters():
        param.requires_grad=False
    
    SupLoss = SegLoss()
    SemiLoss=CEWeightLoss()
    ConLoss=ConsisLoss()
    
    train_params = [{'params': student.get_1x_lr_params(), 'lr': args.lr_init*0.1},
                    {'params': student.get_10x_lr_params(), 'lr': args.lr_init}]
    
    optimizer = torch.optim.Adam(train_params,weight_decay=5e-4) 
    
    memory_pixel=PixelFeatureMemory(num_label, num_classes, num_per_class=256)
    memory_object=ObjectFeatureMemory(num_classes, num_per_class=256)
    
    for i_iter in range(int(args.num_iter)):
        if epoch % lr_adjust_epoch==0 and epoch!=0:
            adjust_learning_rate(optimizer)
            
        #Labeled sample
        try:
            batch_label = next(trainloader_label_iter)
            epoch_flag=False
        except:  
            trainloader_label_iter = iter(trainloader_label)
            batch_label = next(trainloader_label_iter)
            epoch = epoch + 1
            epoch_flag=True
        
        #imgs:[B,3,H,W] labels:[B,H,W] x_dist:[B,2]
        imgs_center,labels,imgs_left,imgs_right,x_dist,y_dist=batch_label
        imgs_center,labels,imgs_left,imgs_right,x_dist,y_dist=imgs_center.cuda(),labels.cuda(),imgs_left.cuda(),imgs_right.cuda(),x_dist.cuda(),y_dist.cuda()
        
        #Disparity inference
        with torch.no_grad():
            feat_l,feat_ld2,_ = disp_FE(imgs_left)
            feat_r,feat_rd2,_ = disp_FE(imgs_right)
            disp_l,disp_r,_,_,= disp_match(feat_ld2,feat_l,feat_rd2,feat_r)
            disp_l=disp_l/4  #[B,1,H,W]
            disp_r=disp_r/4  #[B,1,H,W]
            labels_new=labels.unsqueeze(dim=1).float()  #[B,1,H,W]
            x_dist_l=x_dist[:,0:1].unsqueeze(dim=2).unsqueeze(dim=3) #[B,1,1,1]
            x_dist_r=x_dist[:,1:].unsqueeze(dim=2).unsqueeze(dim=3) #[B,1,1,1]
            y_dist_l=y_dist[:,0:1].unsqueeze(dim=2).unsqueeze(dim=3) #[B,1,1,1]
            y_dist_r=y_dist[:,1:].unsqueeze(dim=2).unsqueeze(dim=3) #[B,1,1,1]
            
            #Pseudo-labels generated by disparity
            pseudo_labels_l=Warp(labels_new,disp_l,x_dist_l,y_dist_l,interpolation='nearest',cuda=True)  #[B,1,H,W]
            imgs_left_warp=Warp(imgs_center,disp_l,x_dist_l,y_dist_l,interpolation='bilinear',cuda=True)  #[B,3,H,W]
            error_left=torch.mean(torch.abs(imgs_left_warp-imgs_left),dim=1)  #[B,H,W]
            weight_left=torch.exp(-error_left/0.5)  #[B,H,W]
            
            pseudo_labels_r=Warp(labels_new,disp_r,x_dist_r,y_dist_r,interpolation='nearest',cuda=True)  #[B,1,H,W]
            imgs_right_warp=Warp(imgs_center,disp_r,x_dist_r,y_dist_r,interpolation='bilinear',cuda=True)  #[B,3,H,W]
            error_right=torch.mean(torch.abs(imgs_right_warp-imgs_right),dim=1)  #[B,H,W]
            weight_right=torch.exp(-error_right/0.5)  #[B,H,W]

        #Build object-level memory bank
        if epoch==0:
            with torch.no_grad():
                labels_onehot=torch.nn.functional.one_hot(labels,num_classes)  #[B,H,W,14]
                labels_onehot=torch.permute(labels_onehot,dims=(0,3,1,2))  #[B,14,H,W]
                smooth=(torch.rand((labels.shape[0],1,labels.shape[1],labels.shape[2]))*0.1).cuda()  #[B,1,H,W]
                labels_smooth = (1.0 - smooth) * labels_onehot + smooth / num_classes  #[B,14,H,W]
                inps,labels_class=gen_class_data(imgs_center,labels_smooth,num_classes)
                
                logits_class,feats_class=class_model(inps)   #[N,14], [N,256]
                probs_class=torch.softmax(logits_class,dim=1)  #[N,14] 
                confs_class,preds_class=torch.max(probs_class,dim=1)  #[N]
                
                mask=((preds_class==labels_class).float()*(confs_class>0.95).float()).bool()  #[N]
                labels_correct=labels_class[mask]  #[n]
                feats_correct=feats_class[mask]  #[n,256]
                confs_correct=confs_class[mask]  #[n]
                memory_object.add_features(feats_correct, labels_correct, confs_correct)

        if epoch_flag==True:
            obj_memory=memory_object.output_features()
            
        #Unlabled samples
        try:
            batch_unlabel = next(trainloader_unlabel_iter)
        except:
            trainloader_unlabel_iter = iter(trainloader_unlabel)
            batch_unlabel = next(trainloader_unlabel_iter)
            
        imgs_left_unlabel,imgs_right_unlabel,imgs_any,x_dist,y_dist =batch_unlabel  #[B,3,H,W]
        imgs_left_unlabel,imgs_right_unlabel,imgs_any,x_dist,y_dist=imgs_left_unlabel.cuda(),imgs_right_unlabel.cuda(),imgs_any.cuda(),x_dist.cuda(),y_dist.cuda()
        
        #Pseudo-labels generated by mean teacher
        with torch.no_grad():
            feat_l,feat_ld2,_ = disp_FE(imgs_left_unlabel)
            feat_r,feat_rd2,_ = disp_FE(imgs_right_unlabel)
            disp_l,disp_r,_,_,= disp_match(feat_ld2,feat_l,feat_rd2,feat_r)  
            disp_l=disp_l/4  #[B,1,H,W]
            disp_r=disp_r/4

            logits_l,_ =teacher(imgs_left_unlabel)
            logits_l=logits_l.detach()
            probs_l=torch.softmax(logits_l,dim=1)  #[B,14,H,W] 
            
            logits_r,_ =teacher(imgs_right_unlabel)
            logits_r=logits_r.detach()
            probs_r=torch.softmax(logits_r,dim=1)  #[B,14,H,W]
            
            logits_any,_ =teacher(imgs_any)
            logits_any=logits_any.detach()
            probs_any=torch.softmax(logits_any,dim=1)  #[B,14,H,W] 
            
            #Merge
            x_dist_l=x_dist[:,0:1].unsqueeze(dim=2).unsqueeze(dim=3) #[B,1,1,1]
            x_dist_r=x_dist[:,1:].unsqueeze(dim=2).unsqueeze(dim=3) #[B,1,1,1]
            y_dist_l=y_dist[:,0:1].unsqueeze(dim=2).unsqueeze(dim=3) #[B,1,1,1]
            y_dist_r=y_dist[:,1:].unsqueeze(dim=2).unsqueeze(dim=3) #[B,1,1,1]
            
            probs_l_warp=Warp(probs_any,disp_l,x_dist_l,y_dist_l,interpolation='nearest',cuda=True)
            imgs_left_unlabel_warp=Warp(imgs_any,disp_l,x_dist_l,y_dist_l,interpolation='bilinear',cuda=True) 
            error_left=torch.mean(torch.abs(imgs_left_unlabel_warp-imgs_left_unlabel),dim=1,keepdim=True)  #[B,1,H,W]
            weight_warp=torch.exp(-error_left/0.5)  #[B,1,H,W]
            weight_warp[weight_warp<0.9]=0
            weight_warp[weight_warp!=0]=0.5
            probs_l_new=weight_warp*probs_l_warp+(1-weight_warp)*probs_l
            conf_l,pseudo_left_unlabel=torch.max(probs_l_new,dim=1)  #[B,H,W]
            conf_l[conf_l<0.3]=0
            weight_left_unlabel = ramp_up(i_iter, RAMP_UP_ITERS)*conf_l  #[B,H,W]

            probs_r_warp=Warp(probs_any,disp_r,x_dist_r,y_dist_r,interpolation='nearest',cuda=True)
            imgs_right_unlabel_warp=Warp(imgs_any,disp_r,x_dist_r,y_dist_r,interpolation='bilinear',cuda=True) 
            error_right=torch.mean(torch.abs(imgs_right_unlabel_warp-imgs_right_unlabel),dim=1,keepdim=True)  #[B,1,H,W]
            weight_warp=torch.exp(-error_right/0.5)  #[B,1,H,W]
            weight_warp[weight_warp<0.9]=0
            weight_warp[weight_warp!=0]=0.5
            probs_r_new=weight_warp*probs_r_warp+(1-weight_warp)*probs_r
            conf_r,pseudo_right_unlabel=torch.max(probs_r_new,dim=1)  #[B,H,W]
            conf_r[conf_r<0.3]=0
            weight_right_unlabel = ramp_up(i_iter, RAMP_UP_ITERS)*conf_r  #[B,H,W]

            m=imgs_left_unlabel.shape[0]
            idx=torch.randperm(m)
            imgs_unlabel1=imgs_left_unlabel[idx][:m//2]  #[B/2,3,H,W]
            disp1=disp_l[idx][:m//2]  #[B/2,1,H,W]
            pseudo_unlabel1=pseudo_left_unlabel[idx][:m//2]  #[B/2,H,W]
            weight_unlabel1=weight_left_unlabel[idx][:m//2]  #[B/2,H,W]

            imgs_unlabel2=imgs_right_unlabel[idx][m//2:]  #[B/2,3,H,W]
            disp2=disp_r[idx][m//2:]  #[B/2,1,H,W]
            pseudo_unlabel2=pseudo_right_unlabel[idx][m//2:]  #[B/2,H,W]
            weight_unlabel2=weight_right_unlabel[idx][m//2:]  #[B/2,H,W]

            imgs_unlabel=torch.cat([imgs_unlabel1,imgs_unlabel2],dim=0)  #[B,3,H,W]
            disp=torch.cat([disp1,disp2],dim=0)  #[B,1,H,W] 
            pseudo_unlabel=torch.cat([pseudo_unlabel1,pseudo_unlabel2],dim=0)  #[B,H,W]
            weight_unlabel=torch.cat([weight_unlabel1,weight_unlabel2],dim=0)  #[B,H,W]
        
        #Build pixel-level memory bank with labeled data
        if i_iter>RAMP_UP_ITERS:
            with torch.no_grad():
                logits_t,feats_t=teacher(imgs_center)
                
                probs_t=torch.softmax(logits_t,dim=1)  #[B,14,H,W] 
                confs_t,preds_t=torch.max(probs_t,dim=1)  #[B,H,W] 
                
                #Correct prediction of teacher
                mask=((preds_t==labels).float()*(confs_t>0.95).float()).bool()  #[B,H,W]
                feats_t=feats_t.permute(0,2,3,1)  #[B,H,W,C]
                labels_correct=labels[mask]  #[N]
                feats_correct=feats_t[mask,...]  #[N,C]
                confs_correct=confs_t[mask]  #[N]
                proj_feats_correct=teacher.projection_head(feats_correct)  #[N,C]

                #Update memory bank
                memory_pixel.add_features(proj_feats_correct, labels_correct, confs_correct, args.batch_size)
                    
        student.train()
        optimizer.zero_grad()
        
        #Supervised loss
        imgs_center_aug,labels_aug=WeakAugment(imgs_center,labels)
        logits_label,feats_label=student(imgs_center_aug)  #[B,14,H,W]

        loss=0
        sup_loss=SupLoss(logits_label,labels_aug)
        loss=loss+sup_loss
        
        #Semi-supervised loss based on disparity
        #Get B samples randomly
        m=imgs_left.shape[0]
        idx=torch.randperm(m)
        imgs1=imgs_left[idx][:m//2]  #[B/2,3,H,W]
        pseudo1=pseudo_labels_l[idx][:m//2]  #[B/2,H,W]
        weight1=weight_left[idx][:m//2]  #[B/2,H,W]

        imgs2=imgs_right[idx][m//2:]  #[B/2,3,H,W]
        pseudo2=pseudo_labels_r[idx][m//2:]  #[B/2,H,W]
        weight2=weight_right[idx][m//2:]  #[B/2,H,W]

        imgs_lr=torch.cat([imgs1,imgs2],dim=0)  #[B,3,H,W]
        pseudo_labels_lr=torch.cat([pseudo1,pseudo2],dim=0).squeeze(dim=1).long()  #[B,H,W]
        weight_lr=torch.cat([weight1,weight2],dim=0)  #[B,H,W]
        
        imgs_lr_aug=torch.zeros_like(imgs_lr)  #[B,3,H,W]
        for i in range(imgs_lr.shape[0]):
            imgs_lr_aug[i]=StrongAugment(imgs_lr[i])
        pseudo_labels_weight=torch.cat([pseudo_labels_lr.unsqueeze(dim=1),weight_lr.unsqueeze(dim=1)],dim=1)  #[B,2,H,W]
        imgs_lr_aug,pseudo_labels_weight_aug=WeakAugment(imgs_lr_aug,pseudo_labels_weight)
              
        logits,_=student(imgs_lr_aug) #[B,14,H,W]
        
        pseudo_labels_aug=pseudo_labels_weight_aug[:,0,:,:].long()
        weight_aug=pseudo_labels_weight_aug[:,1,:,:]
        semi_loss=SemiLoss(logits,pseudo_labels_aug,weight_aug)
        loss=loss+semi_loss
        print('Weighted CE loss-disparity: {}'.format(semi_loss.item()))
        
        #Semi-supervised loss based on mean teacher
        imgs_unlabel_aug=torch.zeros_like(imgs_unlabel)  #[B,3,H,W]
        for i in range(imgs_unlabel.shape[0]):
            imgs_unlabel_aug[i]=StrongAugment(imgs_unlabel[i])
         
        logits_unlabel,feats_unlabel=student(imgs_unlabel_aug) #[B,14,H,W]
        
        semi_loss=SemiLoss(logits_unlabel,pseudo_unlabel,weight_unlabel)
        loss=loss+semi_loss
        print('Weighted CE loss-teacher: {}'.format(semi_loss.item()))

        con_loss=ConLoss(logits_unlabel,disp)
        loss=loss+0.1*con_loss
        print('Consistency loss: {}'.format(con_loss.item()))
        
        #Contrastive loss
        if i_iter>RAMP_UP_ITERS:
            #Pixel-level for labeled data
            probs_label=torch.softmax(logits_label,dim=1)
            confs_label,preds_label=torch.max(probs_label,dim=1)

            #False predictions
            mask_false=(preds_label!=labels_aug)  #[B,H,W]
            labels_false=labels_aug[mask_false]  #[N]
            feats_label=feats_label.permute(0,2,3,1)  #[B,H,W,C]
            feats_false=feats_label[mask_false,...]  #[N,C]

            #Correct predictions but with low confidence
            mask_low=((preds_label==labels_aug).float()*(confs_label<=0.6).float()).bool()  #[B,H,W]
            labels_low=labels_aug[mask_low]   #[N']
            feats_low=feats_label[mask_low,...]   #[N',C]
            labels_false=torch.cat([labels_false,labels_low],dim=0)  #[N+N']
            feats_false=torch.cat([feats_false,feats_low],dim=0)  #[N+N',C]
            proj_feats_false=student.projection_head(feats_false)   #[N,C]
            contr_loss_label=ContrastiveLoss(proj_feats_false, labels_false, memory_pixel.memory, num_classes=14)
            loss=loss+contr_loss_label*0.1
            print('Pixel contrastive loss-label: {}'.format(contr_loss_label.item()))
            
            #Pixel-level for unlabeled data
            probs_unlabel=torch.softmax(logits_unlabel,dim=1)
            _,preds_unlabel=torch.max(probs_unlabel,dim=1)
            mask_false=((preds_unlabel!=pseudo_unlabel).float()*(weight_unlabel>0.5).float()).bool()
            pseudo_false=pseudo_unlabel[mask_false]  #[N]
            feats_unlabel=feats_unlabel.permute(0,2,3,1)  #[B,H,W,C]
            feats_false=feats_unlabel[mask_false,...]  #[N,C]
            proj_feats_false=student.projection_head(feats_false)   #[N,C]
            contr_loss_unlabel=ContrastiveLoss(proj_feats_false, pseudo_false, memory_pixel.memory, num_classes=14)
            loss=loss+contr_loss_unlabel*0.1
            print('Pixel contrastive loss-unlabel: {}'.format(contr_loss_unlabel.item()))

            #Entropy loss
            mask_entropy=(weight_unlabel<0.5)  #[B,H,W]
            probs_unlabel=probs_unlabel.permute(0,2,3,1)
            probs=probs_unlabel[mask_entropy,...]
            entropy_loss=EntropyLoss(probs)
            loss=loss+entropy_loss*0.01
            print('Entropy loss-unlabel: {}'.format(entropy_loss.item()))

            #Object-level for unlabeled data
            if epoch>0:              
                probs_s=torch.softmax(logits_unlabel,dim=1)  #[B,14,H,W]
                inps,labels_class=gen_class_data(imgs_unlabel,probs_s,num_classes)
                _,feats_class=class_model(inps)   #[N,256]
                    
                contr_obj_loss=ContrastiveLoss(feats_class,labels_class,obj_memory,num_classes)
                loss=loss+contr_obj_loss*0.1
                print('Object contrastive loss-unlabel: {}'.format(contr_obj_loss.item()))
            
        loss.backward()
        optimizer.step()
        print('Epoch: %i, iter: %i, train_loss: %f' %(epoch,i_iter,loss.item()))
        print('')
        
        #Update teacher
        m = 1 - (1 - 0.995) * (math.cos(math.pi * i_iter / args.num_iter) + 1) / 2
        teacher=teacher_update(student,teacher,m,i_iter)
        
        #Validation
        if epoch_flag==True:
            student.eval()
            with torch.no_grad():
                miou_list=[]
                for imgs,labels in testloader:
                    imgs=imgs.cuda()  #[B,9,3,H,W]
                    logits,_=student(imgs) #[B,14,H,W]
                    preds=torch.argmax(logits,dim=1)  #[B,H,W]
                    
                    preds=preds.cpu().numpy()
                    labels=labels.numpy()
                    B=preds.shape[0]
                    for i in range(B):
                        p=preds[i]
                        l=labels[i]
                        miou=IOU_mean(p,l,ignore_class=3)
                        miou_list.append(miou)
    
                ave_miou=np.mean(miou_list)
                print('Epoch {}, mIOU {}'.format(epoch,ave_miou))
                print('')
            
            #Write results to a txt file
            f=open(os.path.join(args.save_root,'results.txt'),'a',encoding='utf-8')
            f.write(str(ave_miou))
            f.write('\n')
            f.close()
            
            #Save models            
            if epoch<=args.save_num:
                torch.save({'student':student.state_dict(),'teacher':teacher.state_dict()},os.path.join(args.save_root,'model%s.pth'%str(epoch)))
                #torch.save(teacher.state_dict(),os.path.join(args.save_root,'model%s.pth'%str(epoch)))
                miou_dict[str(epoch)]=ave_miou
                epoch_dict[str(epoch)]=epoch
            else:
                if ave_miou>min(miou_dict.values()):
                    torch.save({'student':student.state_dict(),'teacher':teacher.state_dict()},os.path.join(args.save_root,'model%s.pth'%(min(miou_dict,key=lambda x: miou_dict[x]))))
                    #torch.save(teacher.state_dict(),os.path.join(args.save_root,'model%s.pth'%(min(miou_dict,key=lambda x: miou_dict[x]))))
                    epoch_dict[min(miou_dict,key=lambda x: miou_dict[x])]=epoch
                    miou_dict[min(miou_dict,key=lambda x: miou_dict[x])]=ave_miou
                    
            if ave_miou>best_miou:
                best_miou=ave_miou
                best_epoch=epoch
            print('Best Mean IOU {}, epoch {}'.format(best_miou,best_epoch))
            print('Epoch {}'.format(epoch_dict))
            print('Mean IOU {}'.format(miou_dict))
            print('')
    

if __name__=='__main__':  
    
    parser = argparse.ArgumentParser(description='Semi-supervised Semantic Segmentation')
    parser.add_argument('--data_root', default='', type=str)
    parser.add_argument('--labeled_img_path', default='', type=str)
    parser.add_argument('--unlabeled_img_path', default='', type=str)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--lr_init', default=1e-3, type=float)
    parser.add_argument('--num_iter',default=5e4,type=int)
    parser.add_argument('--save_num', default=10, type=int, help='number of saved models')

    parser.add_argument('--save_root',default='/mnt/disk/sszhang/save_models/DeepLabv3+50',type=str)
    parser.add_argument('--disp_model_path',default='/mnt/disk/sszhang/save_models/StereoNet/model.pth',type=str)
    parser.add_argument('--class_model_path',default='/mnt/disk/sszhang/save_models/ClassNet/model.pth',type=str)
    
    parser.add_argument('--pre_train',default=False,type=bool)
    parser.add_argument('--model_path',default='/mnt/disk/sszhang/save_models/DeepLabv3+50/model.pth',type=str)
    args = parser.parse_known_args()[0]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    train(args)
