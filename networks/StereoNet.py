import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from blocks import ResBlock, ResBlock3D, UpSkip3D
sys.path.append('/home/sszhang/codes')
from common.ImgProcess import Warp


class Feature_extractor(nn.Module):
    def __init__(self,in_channel,base_channel):
        super(Feature_extractor,self).__init__()
        num_channels=[base_channel,base_channel*2,base_channel*4,base_channel*8]
        self.conv1=nn.Sequential(nn.Conv2d(in_channel,num_channels[0],kernel_size=3,stride=1,padding=1),
                                 nn.GroupNorm(num_groups=8, num_channels=num_channels[0]),
                                 nn.LeakyReLU(inplace=True))
        self.block11=ResBlock(num_channels[0],num_channels[1],stride=1)
        self.block12=ResBlock(num_channels[1],num_channels[1],stride=1)
        self.block13=ResBlock(num_channels[1],num_channels[1],stride=1)
        
        self.block21=ResBlock(num_channels[1],num_channels[2],stride=2)
        self.block22=ResBlock(num_channels[2],num_channels[2],stride=1)
        self.block23=ResBlock(num_channels[2],num_channels[2],stride=1)
        
        self.block31=ResBlock(num_channels[2],num_channels[3],stride=2)
        self.block32=ResBlock(num_channels[3],num_channels[3],stride=1)
        self.block33=ResBlock(num_channels[3],num_channels[3],stride=1)
        
        self.up21=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear'),
                                nn.Conv2d(num_channels[2],num_channels[1],kernel_size=1,stride=1,padding=0))
        
        self.up31=nn.Sequential(nn.Upsample(scale_factor=4,mode='bilinear'),
                                nn.Conv2d(num_channels[3],num_channels[1],kernel_size=1,stride=1,padding=0))
        self.fuse1=ResBlock(num_channels[1],num_channels[1],stride=1)
        
        self.down12=nn.Conv2d(num_channels[1],num_channels[2],kernel_size=1,stride=1,padding=0)
        self.up32=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear'),
                                nn.Conv2d(num_channels[3],num_channels[2],kernel_size=1,stride=1,padding=0))
        self.fuse2=ResBlock(num_channels[2],num_channels[2],stride=1)
        
        
    def forward(self,inp):
        feat=self.conv1(inp)  #[B,16,H,W]
        feat=self.block11(feat)  #[B,32,H,W]
        feat_d2=self.block21(feat)  #[B,64,H/2,W/2]
        feat_d4=self.block31(feat_d2)  #[B,128,H/4,W/4]
        
        feat=self.block12(feat)
        feat=self.block13(feat)  #[B,32,H,W]  
        
        feat_d2=self.block22(feat_d2)
        feat_d2=self.block23(feat_d2)  #[B,64,H/2,W/2]
        
        feat_d4=self.block32(feat_d4)
        feat_d4=self.block33(feat_d4)  #[B,128,H/4,W/4]
        
        feat_out=self.fuse1(feat+self.up21(feat_d2)+self.up31(feat_d4))
        temp=nn.functional.interpolate(feat,scale_factor=0.5,mode='bilinear')
        feat_d2_out=self.fuse2(feat_d2+self.down12(temp)+self.up32(feat_d4))
        
        return feat_out,feat_d2_out,feat_d4
    

def Group_correlation(feat1, feat2, num_groups):
    B, C, H, W = feat1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups

    cost = (feat1 * feat2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


class Cost_volume_left(nn.Module):
    def __init__(self,disp_scale,group_num):  
        super(Cost_volume_left,self).__init__()
        self.disp_scale=disp_scale
        self.group_num=group_num
        
    def forward(self,feat_left,feat_right,disp_init):
        disp_num=len(self.disp_scale)
        B,C,H,W=feat_left.size()
        cost_volume=Variable(torch.FloatTensor(B,self.group_num,disp_num,H,W).zero_()).cuda()
        k=0

        for disp in self.disp_scale:
            disp=disp_init+disp
            feat_warp=Warp(feat_right,disp,x_dist=-1,y_dist=0,cuda=True)
            cost=Group_correlation(feat_left, feat_warp, self.group_num)
            cost_volume[:,:,k,:,:]=cost
            k+=1
            
        cost_volume=cost_volume.contiguous() 
        
        return cost_volume
    
    
class Cost_volume_right(nn.Module):
    def __init__(self,disp_scale,group_num):  
        super(Cost_volume_right,self).__init__()
        self.disp_scale=disp_scale
        self.group_num=group_num
        
    def forward(self,feat_left,feat_right,disp_init):
        disp_num=len(self.disp_scale)
        B,C,H,W=feat_left.size()
        cost_volume=Variable(torch.FloatTensor(B,self.group_num,disp_num,H,W).zero_()).cuda()
        k=0

        for disp in self.disp_scale:
            disp=disp_init+disp
            feat_warp=Warp(feat_left,disp,x_dist=1,y_dist=0,cuda=True)
            cost=Group_correlation(feat_right, feat_warp, self.group_num)
            cost_volume[:,:,k,:,:]=cost
            k+=1
            
        cost_volume=cost_volume.contiguous()  #[B,G,D,H,W]
        
        return cost_volume
    
    
#Cost filter
class Cost_filter(nn.Module):
    def __init__(self,in_channels):
        super(Cost_filter,self).__init__()
        self.block1=ResBlock3D(in_channels,in_channels,stride=2)
        self.block2=ResBlock3D(in_channels,in_channels*2,stride=2)
        self.block3=ResBlock3D(in_channels*2,in_channels*2,stride=1)
        
        self.up1=UpSkip3D(in_channels*2,in_channels,mode='sum')
        self.up_block1=ResBlock3D(in_channels,in_channels,stride=1)
        
        self.up2=UpSkip3D(in_channels,in_channels,mode='sum')
        self.up_block2=ResBlock3D(in_channels,in_channels,stride=1)
        
        
    def forward(self,x):
        #inp:[N,G,D,H,W]
        map1=x
        map2=self.block1(map1)  #[B,G,D,H/2,W/2]
        map3=self.block2(map2)  #[B,2G,D,H/4,W/4]
        map3=self.block3(map3)  #[B,2G,D,H/4,W/4]
        
        map2=self.up1(map3,map2) #[B,G,D,H/2,W/2]
        map2=self.up_block1(map2) #[B,G,D,H/2,W/2]
        map1=self.up2(map2,map1) #[B,G,D,H,W]
        out=self.up_block2(map1) #[B,G,D,H,W]
        
        return out
    
    
class Regression(nn.Module):
    def __init__(self,in_channels,disp_scale):
        super(Regression,self).__init__()
        self.conv1=nn.Sequential(nn.Conv3d(in_channels,in_channels//4,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                                 nn.GroupNorm(num_groups=min(8,in_channels//4), num_channels=in_channels//4),
                                 nn.LeakyReLU(inplace=True))
        self.conv2=nn.Sequential(nn.Conv3d(in_channels//4,1,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
                                 nn.LeakyReLU(inplace=True))
        self.disp_scale=disp_scale
        
    def forward(self,x):
        x=self.conv1(x)  #[B,G/4,D,H,W]
        x=self.conv2(x)  #[B,1,D,H,W]
        x=torch.squeeze(x,dim=1)  #[B,D,H,W]
        
        prob=F.softmax(x,dim=1)
        disp_scale=torch.Tensor(np.reshape(self.disp_scale,[1,self.disp_scale.shape[0],1,1])).cuda()
        disp = torch.sum(prob*disp_scale.data, dim=1, keepdim=True)  #[B,1,H,W]
        
        return disp
    

class StereoMatch(nn.Module):
    def __init__(self,filter_num,scale_dict,group_num):
        #scale_dict={'d2':np.linspace(-1,3,20),'full':np.linspace(-0.2,0.6,20)}
        super(StereoMatch,self).__init__()
        self.cost_volume_ld2=Cost_volume_left(scale_dict['d2'],group_num)
        self.cost_volume_rd2=Cost_volume_right(scale_dict['d2'],group_num)
        self.cost_volume_l=Cost_volume_left(scale_dict['full'],group_num)
        self.cost_volume_r=Cost_volume_right(scale_dict['full'],group_num)
        
        filter_list=[]
        for i in range(filter_num):
            cost_filter=Cost_filter(group_num)
            filter_list.append(cost_filter)
        self.filter_list=nn.ModuleList(filter_list)
        
        self.regression_d2=Regression(group_num,scale_dict['d2'])
        self.regression=Regression(group_num,scale_dict['full'])
        
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear')
        
    def forward(self,feat_ld2,feat_l,feat_rd2,feat_r):
        
        #1/2 scale
        cost_ld2=self.cost_volume_ld2(feat_ld2,feat_rd2,disp_init=0)  #[B,G,D,H/2,W/2]
        cost_rd2=self.cost_volume_rd2(feat_ld2,feat_rd2,disp_init=0)  #[B,G,D,H/2,W/2]
        
        for cost_filter in self.filter_list:
            cost_ld2=cost_filter(cost_ld2)   #[B,G,D,H/2,W/2]
            cost_rd2=cost_filter(cost_rd2)   #[B,G,D,H/2,W/2]
            
        disp_ld2=self.regression_d2(cost_ld2) #[B,1,H/2,W/2]
        disp_rd2=self.regression_d2(cost_rd2)  #[B,1,H/2,W/2]
        
        #Full scale
        disp_l=self.upsample(disp_ld2)*2
        disp_r=self.upsample(disp_rd2)*2
        
        cost_l=self.cost_volume_l(feat_l,feat_r,disp_init=disp_l)  #[B,G,D,H,W]
        cost_r=self.cost_volume_r(feat_l,feat_r,disp_init=disp_r)  #[B,G,D,H,W]
        
        for cost_filter in self.filter_list:
            cost_l=cost_filter(cost_l)   #[B,G,D,H,W]
            cost_r=cost_filter(cost_r)   #[B,G,D,H,W]
            
        disp_l=self.regression(cost_l)+disp_l  #[B,1,H,W]
        disp_r=self.regression(cost_r)+disp_r  #[B,1,H,W]
        
        return disp_l,disp_r,disp_ld2,disp_rd2
        
           
        
        
        


