import numpy as np
from skimage.metrics import structural_similarity


def PSNR(img1,img2,data_range):
    mse = ((img1 - img2) ** 2).mean()
    psnr = 10 * np.log10(np.power(data_range,2) / (mse+1e-6))
    return psnr

def SSIM(img1,img2,data_range,channel_first=True):
    if channel_first:
        c=img1.shape[0]
        total_ssim=0
        for i in range(c):
            img1_c=img1[i]
            img2_c=img2[i]
            ssim=structural_similarity(img1_c,img2_c,data_range=data_range)
            total_ssim+=ssim
        ssim=total_ssim/c
    else:
        c=img1.shape[-1]
        total_ssim=0
        for i in range(c):
            img1_c=img1[...,i]
            img2_c=img2[...,i]
            ssim=structural_similarity(img1_c,img2_c,data_range=data_range)
            total_ssim+=ssim
        ssim=total_ssim/c
    return ssim


def IOU_mean(pred,target,ignore_class):
    iou_list=[]
    cls_list=np.unique(target)
    for cla in cls_list:
        if cla!=ignore_class:
            p=(pred==cla).astype(int)
            t=(target==cla).astype(int)
            intersection=np.sum(p*t)
            union=np.sum(p)+np.sum(t)-intersection
            iou=intersection/union
            iou_list.append(iou)
    iou_mean=np.mean(iou_list)
    return iou_mean