import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
import itertools
import argparse

from networks.StereoNet import Feature_extractor, StereoMatch
from Dataset import DispUnTrainData, DispValData
from Loss import SelfDispLoss


def adjust_learning_rate(optimizer,lr_init,epoch,lr_step):
    lr = lr_init* (0.9 ** (epoch // lr_step))
    if lr>=5e-5:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-5
        
        
def train(args):
    lr_step=50
    epoch_dict,mse_dict={},{}
    for i in range(1,args.save_num+1):
        epoch_dict[str(i)]=0
        mse_dict[str(i)]=0
    best_mse=float('inf')
    best_epoch=0
    
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root,exist_ok=True)
            
    train_set=DispUnTrainData(os.path.join(args.data_root,'Syn/train'),args.crop_size)
    val_set=DispValData(os.path.join(args.data_root,'Syn/val'))
    
    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(dataset=val_set,batch_size=args.batch_size,shuffle=False)
    
    FE=Feature_extractor(in_channel=3,base_channel=16)
    print('FE parameters: {}'.format(sum(param.numel() for param in FE.parameters())))
    
    scale_dict={'d2':np.linspace(-1,3,20),'full':np.linspace(-0.2,0.6,20)}
    match_model=StereoMatch(filter_num=3,scale_dict=scale_dict,group_num=16)
    print('Match model parameters: {}'.format(sum(param.numel() for param in match_model.parameters())))
    
    criterion=SelfDispLoss()
    
    FE.cuda() 
    match_model.cuda()
    
    if args.pre_train:
        checkpoint=torch.load(args.model_path)
        FE.load_state_dict(checkpoint['FE'])
        match_model.load_state_dict(checkpoint['Match'])
          
    optimizer = torch.optim.Adam(itertools.chain(FE.parameters(),match_model.parameters()),lr=args.lr_init)
    
    for epoch in range(args.max_epoch):
        FE.train()
        match_model.train()
        if epoch % lr_step==0 and epoch!=0:
            adjust_learning_rate(optimizer,args.lr_init,epoch,lr_step)
        
        for idx,(img_l,img_r,img_ld2,img_rd2) in enumerate(train_loader):
            img_l,img_r,img_ld2,img_rd2=Variable(img_l).cuda(),Variable(img_r).cuda(),Variable(img_ld2).cuda(),Variable(img_rd2).cuda()
            optimizer.zero_grad()
        
            feat_l,feat_ld2 = FE(img_l)
            feat_r,feat_rd2 = FE(img_r)
            disp_l,disp_r,disp_ld2,disp_rd2=match_model(feat_ld2,feat_l,feat_rd2,feat_r)
    
            loss=criterion(disp_l,disp_r,disp_ld2,disp_rd2,img_l,img_r,img_ld2,img_rd2)
            loss.backward()
            optimizer.step()
            print('Epoch: %i, batch_idx: %i, train_loss: %f' %(epoch,idx,loss.item()))
            print('')
            
        FE.eval()
        match_model.eval()
        with torch.no_grad():
            mse_list=[]
            
            for img_l,img_r,disp_l_gt,disp_r_gt in val_loader:
                img_l,img_r=img_l.cuda(),img_r.cuda()              
                feat_l,feat_ld2 = FE(img_l)
                feat_r,feat_rd2 = FE(img_r)
                disp_l,disp_r,_,_,=match_model(feat_ld2,feat_l,feat_rd2,feat_r)
                disp_l=disp_l/4
                disp_r=disp_r/4

                disp_l=disp_l.cpu().numpy().squeeze()
                disp_r=disp_r.cpu().numpy().squeeze()
                disp_l_gt=disp_l_gt.numpy()
                disp_r_gt=disp_r_gt.numpy()
                
                B=disp_l.shape[0]
                for i in range(B):
                    disp_l_one=disp_l[i]
                    disp_l_gt_one=disp_l_gt[i]
                    mse_l=np.mean(np.power((disp_l_one-disp_l_gt_one),2))
                    mse_list.append(mse_l)

                    disp_r_one=disp_r[i]
                    disp_r_gt_one=disp_r_gt[i]
                    mse_r=np.mean(np.power((disp_r_one-disp_r_gt_one),2))
                    mse_list.append(mse_r)
                        
            ave_mse=np.mean(mse_list)
            print('Epoch {}, average MSE {}'.format(epoch,ave_mse))
            print('')
    
        #save models            
        if epoch<args.save_num:
            torch.save({'FE':FE.state_dict(),'Match':match_model.state_dict()},os.path.join(args.save_root,'model%s.pth'%str(epoch+1)))
            mse_dict[str(epoch+1)]=ave_mse
            epoch_dict[str(epoch+1)]=epoch
        else:
            if ave_mse<max(mse_dict.values()):
                torch.save({'FE':FE.state_dict(),'Match':match_model.state_dict()},
                            os.path.join(args.save_root,'model%s.pth'%(max(mse_dict,key=lambda x: mse_dict[x]))))
                epoch_dict[max(mse_dict,key=lambda x: mse_dict[x])]=epoch
                mse_dict[max(mse_dict,key=lambda x: mse_dict[x])]=ave_mse
                
        if ave_mse<best_mse:
            best_mse=ave_mse
            best_epoch=epoch
        print('Best MSE {}, epoch {}'.format(best_mse,best_epoch))
        print('Epoch {}'.format(epoch_dict))
        print('MSE {}'.format(mse_dict))
        print('')


if __name__=='__main__':  
    
    parser = argparse.ArgumentParser(description='Disparity')
    parser.add_argument('--data_root', default='', type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr_init', default=5e-4, type=float)
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--max_epoch',default=200,type=int)
    parser.add_argument('--save_num', default=10, type=int, help='number of saved models')
    parser.add_argument('--save_root',default='/home/sszhang/save_models/StereoNet',type=str)
    parser.add_argument('--pre_train',default=False,type=bool)
    parser.add_argument('--model_path',default='/home/sszhang/save_models/StereoNet/model7.pth',type=str)
    args = parser.parse_known_args()[0]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    train(args)
