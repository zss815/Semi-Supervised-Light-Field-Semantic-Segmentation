import torch
from torch import nn
import torch.nn.functional as F

from common.losses import SSIM, Smooth_loss
from common.ImgProcess import Warp


class SelfDispLoss(nn.Module):
    def __init__(self):
        super(SelfDispLoss,self).__init__()
        self.l1_loss=nn.L1Loss()
        self.ssim=SSIM(window_size=11,size_average=True)
    
    def forward(self,disp_l,disp_r,disp_ld2,disp_rd2,img_l,img_r,img_ld2,img_rd2):
        img_ld2_warp=Warp(img_rd2,disp_ld2,x_dist=-1,y_dist=0,cuda=True)
        img_rd2_warp=Warp(img_ld2,disp_rd2,x_dist=1,y_dist=0,cuda=True)
        pm_d2_loss=self.l1_loss(img_ld2_warp,img_ld2)+self.l1_loss(img_rd2_warp,img_rd2)
        smooth_d2_loss=Smooth_loss(disp_ld2,img_ld2,gamma=50)+Smooth_loss(disp_rd2,img_rd2,gamma=50)
        print('pm_loss d2: {}, smooth_loss d2: {}'.format(pm_d2_loss,smooth_d2_loss))
        
        img_l_warp=Warp(img_r,disp_l,x_dist=-1,y_dist=0,cuda=True)
        img_r_warp=Warp(img_l,disp_r,x_dist=1,y_dist=0,cuda=True)
        pm_loss=self.l1_loss(img_l_warp,img_l)+self.l1_loss(img_r_warp,img_r)
        smooth_loss=Smooth_loss(disp_l,img_l,gamma=50)+Smooth_loss(disp_r,img_r,gamma=50)
        ssim_loss=1-(self.ssim(img_l_warp,img_l)+self.ssim(img_r_warp,img_r))/2
        print('pm_loss: {}, ssim_loss: {}, smooth_loss: {}'.format(pm_loss,ssim_loss,smooth_loss))
        
        full_loss=pm_d2_loss+0.1*smooth_d2_loss+pm_loss+ssim_loss+0.1*smooth_loss

        monitor_loss=5*pm_loss+ssim_loss
        
        return full_loss,monitor_loss

        

def DiceLoss(prob,target):
    smooth=1e-5
    total=0
    for i in range(prob.shape[1]):
        p=prob[:,i,:,:]
        t=target[:,i,:,:]
        num=2*torch.sum(p*t,dim=(1,2))+smooth
        den=torch.sum(p,dim=(1,2))+torch.sum(t,dim=(1,2))+smooth
        loss=1-torch.mean(num/den)
        total+=loss
    dice_loss=total/prob.shape[1]
    return dice_loss
            

class CEWeightLoss(nn.Module):
    def __init__(self):
        super(CEWeightLoss,self).__init__()
        self.ce_loss=nn.CrossEntropyLoss(reduction='none')
    
    def forward(self,logit,label,weight):
        loss=self.ce_loss(logit,label) #[B,H,W]
        loss=torch.sum(loss*weight)/(torch.sum((weight!=0).float())+1e-5)
        return loss


class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss,self).__init__()
        self.ce_loss=nn.CrossEntropyLoss()
    
    def forward(self,logit,label):
        
        ce_loss=self.ce_loss(logit,label)
        
        prob=nn.Softmax(dim=1)(logit)
        target=nn.functional.one_hot(label,num_classes=14)
        target=torch.permute(target,dims=(0,3,1,2))
        dice_loss=DiceLoss(prob,target)
     
        print('CE loss: {}, dice loss: {}'.format(ce_loss,dice_loss))
        
        full_loss=dice_loss+ce_loss
        
        return full_loss


#Contrastive loss
def ContrastiveLoss(features, labels, memory, num_classes):
    """
    Args:
        features: [N,32]
        labels: [N] 
        memory: memory bank [List]
    Returns:
        returns the contrastive loss between features vectors from [features] and from [memory] in a class-wise fashion.
    """

    loss = 0

    for c in range(num_classes):
        mask_c = labels == c
        features_c = features[mask_c,:]  #[n,32]
        memory_c = memory[c]  #[m,32]

        if memory_c is not None and features_c.shape[0] > 1 and memory_c.shape[0] > 1:

            memory_c = torch.from_numpy(memory_c).cuda()

            # L2 normalize vectors
            memory_c_norm = F.normalize(memory_c, dim=1)  #[m,32]
            features_c_norm = F.normalize(features_c, dim=1)  #[n,32]

            # compute similarity. All elements with all elements
            similarity = torch.mm(features_c_norm, memory_c_norm.transpose(1, 0))  # [n,m]
            distance = 1 - similarity  # values between [0, 2] where 0 means same vectors

            loss = loss + distance.mean()

    return loss/num_classes
        
#Entropy loss
def EntropyLoss(probs):
    """
    Args:
        probs: [N,C]
    """
    loss= torch.mul(probs, torch.log(probs + 1e-5))
    loss = -torch.sum(loss, dim=1)
    return torch.mean(loss)   


#Pixel-wise cosine similarity
def CosSim(feat1,feat2):
    #inp: [B,C,H,W]
    inner=torch.sum(feat1*feat2,dim=1)  #[B,H,W]
    mag1=torch.sum(torch.square(feat1),dim=1).sqrt()  #[B,H,W]
    mag2=torch.sum(torch.square(feat2),dim=1).sqrt()  #[B,H,W]
    cos=inner/(mag1*mag2) #[B,H,W]
    return cos


class ConsisLoss(nn.Module):
    def __init__(self):
        super(ConsisLoss,self).__init__()
        self.softmax=nn.Softmax(dim=1)
        self.l1_loss=nn.L1Loss()
        
    def forward(self,logit,disp):
        disp=disp.detach()
        
        prob=self.softmax(logit)
        cos_x=1-CosSim(prob[:, :, :, :-1],prob[:, :, :, 1:]) #[B,H,W-1]
        cos_y=1-CosSim(prob[:, :, :-1, :],prob[:, :, 1:, :]) #[B,H-1,W]
        
        grad_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:]).squeeze(dim=1)  #[B,H,W-1]
        grad_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :]).squeeze(dim=1)  #[B,H-1,W]
        grad_x_norm=torch.zeros_like(grad_x).cuda()
        grad_y_norm=torch.zeros_like(grad_y).cuda()

        for i in range(grad_x.shape[0]):
            gx=grad_x[i]
            gx=(gx-gx.min())/(gx.max()-gx.min())
            grad_x_norm[i]=gx
            gy=grad_y[i]
            gy=(gy-gy.min())/(gy.max()-gy.min())
            grad_y_norm[i]=gy
        
        consis_loss=self.l1_loss(cos_x,grad_x_norm)+self.l1_loss(cos_y,grad_y_norm)
        
        return consis_loss