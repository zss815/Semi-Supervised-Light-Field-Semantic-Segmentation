import torch
import torch.nn as nn
from torch.autograd import Variable


    
def Warp(img_source,disp,x_dist,y_dist,interpolation='bilinear',cuda=True):
    #left+ right- top+ down-
    B, C, H, W = img_source.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    if cuda:
        xx = xx.view(1,1,H,W).repeat(B,1,1,1).cuda()
        yy = yy.view(1,1,H,W).repeat(B,1,1,1).cuda()
    else:
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    warp_grid=Variable(grid)
    warp_grid[:,0:1,:,:]=warp_grid[:,0:1,:,:]+disp*x_dist
    warp_grid[:,1:,:,:]=warp_grid[:,1:,:,:]+disp*y_dist
    
    # scale grid to [-1,1]
    warp_grid[:,0,:,:] = 2.0*warp_grid[:,0,:,:].clone() / max(W-1,1)-1.0
    warp_grid[:,1,:,:] = 2.0*warp_grid[:,1,:,:].clone() / max(H-1,1)-1.0
    
    warp_grid = warp_grid.permute(0,2,3,1)        
    img_warp = nn.functional.grid_sample(img_source,warp_grid,mode=interpolation,padding_mode='border')
    return img_warp