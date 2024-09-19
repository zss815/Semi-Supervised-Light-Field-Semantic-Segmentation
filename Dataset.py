import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import GaussianBlur, Resize, InterpolationMode
import random
from utils.DataAugmentation import StrongAugment, WeakAugment


#Self-supervised disparity training data
class DispUnTrainData(Dataset):
    def __init__(self,data_root,crop_size):
        super(DispUnTrainData,self).__init__()
        self.img_path=[]
        for item in os.listdir(os.path.join(data_root)):
            if not item.startswith('.'):
                self.img_path.append(os.path.join(data_root,item))
        self.crop_size=crop_size
        
    def __getitem__(self,index):
        row=np.random.choice(np.arange(1,10))
        col=np.random.choice(np.arange(1,6))
        name_left=str(row)+'_'+str(col)
        name_right=str(row)+'_'+str(col+4)

        img_left=np.array(Image.open(os.path.join(self.img_path[index],name_left+'.png')))/255  #[H,W,3]
        img_right=np.array(Image.open(os.path.join(self.img_path[index],name_right+'.png')))/255  #[H,W,3]
        img_left=np.transpose(img_left,axes=(2,0,1))   #[3,H,W]
        img_right=np.transpose(img_right,axes=(2,0,1))   #[3,H,W]
        
        #random crop
        H,W=img_left.shape[1],img_left.shape[2]
        h_begin=np.random.randint(H-self.crop_size)
        w_begin=np.random.randint(W-self.crop_size)
        img_left=img_left[:,h_begin:h_begin+self.crop_size,w_begin:w_begin+self.crop_size]
        img_right=img_right[:,h_begin:h_begin+self.crop_size,w_begin:w_begin+self.crop_size]    
        
        img_left=torch.from_numpy(img_left).float()  #[3,h,w]
        img_right=torch.from_numpy(img_right).float() #[3,h,w]
        
        #Downsample
        h,w=img_left.shape[1],img_left.shape[2]
        img_left_blur=GaussianBlur(kernel_size=3,sigma=0.5)(img_left)
        img_left_d2=Resize(size=(h//2,w//2))(img_left_blur)  #[3,h/2,w/2]
        
        img_right_blur=GaussianBlur(kernel_size=3,sigma=0.5)(img_right)
        img_right_d2=Resize(size=(h//2,w//2))(img_right_blur)  #[3,h/2,w/2]
        
        return img_left,img_right,img_left_d2,img_right_d2
            
    def __len__(self):
        return len(self.img_path)


#Disparity validation data           
class DispValData(Dataset):
    def __init__(self,data_root):
        super(DispValData,self).__init__()
        self.img_path=[]
        for item in os.listdir(data_root):
            if not item.startswith('.'):
                if '1_1_disparity.npy' in os.listdir(os.path.join(data_root,item)):
                    self.img_path.append(os.path.join(data_root,item))
        
    def __getitem__(self,index):
        row=5
        col=5
        name_left=str(row)+'_'+str(col)
        name_right=str(row)+'_'+str(col+4)

        img_left=np.array(Image.open(os.path.join(self.img_path[index],name_left+'.png')))/255  #[H,W,3]
        img_right=np.array(Image.open(os.path.join(self.img_path[index],name_right+'.png')))/255  #[H,W,3]
        img_left=np.transpose(img_left,axes=(2,0,1))   #[3,H,W]
        img_right=np.transpose(img_right,axes=(2,0,1))   #[3,H,W]
        img_left=torch.from_numpy(img_left).float()  #[3,H,W]
        img_right=torch.from_numpy(img_right).float() #[3,H,W]
        
        disp_left=np.load(os.path.join(os.path.join(self.img_path[index],name_left+'_disparity.npy')))
        disp_right=np.load(os.path.join(os.path.join(self.img_path[index],name_right+'_disparity.npy')))
        disp_left=torch.from_numpy(disp_left).float()  #[H,W]
        disp_right=torch.from_numpy(disp_right).float()  #[H,W]
        
        return img_left,img_right,disp_left,disp_right
            
    def __len__(self):
        return len(self.img_path)


#Semi-supervised segmentation training data 
class SegLabelData(Dataset):
    def __init__(self,img_path):
        super(SegLabelData,self).__init__()
        self.img_path=img_path
    
    def __getitem__(self,index):
        path=self.img_path[index]
        img_center=np.array(Image.open(os.path.join(path,'5_5.png')))/255  #[H,W,3]
            
        if path.split('/')[-3]=='Real':
            label=np.load(os.path.join(path,'label.npy'))  #[H,W] 
        else:
            label=np.load(os.path.join(path,'5_5_label.npy'))  #[H,W]
            
        row=np.random.choice(np.arange(1,10))
        col_l=np.random.choice(np.arange(1,5))
        col_r=col_l+4
        img_left=np.array(Image.open(os.path.join(path,str(row)+'_'+str(col_l)+'.png')))/255  #[H,W,3]
        img_right=np.array(Image.open(os.path.join(path,str(row)+'_'+str(col_r)+'.png')))/255  #[H,W,3]
        
        img_center=torch.from_numpy(img_center).float().permute(2,0,1)  #[3,H,W]
        label=torch.from_numpy(label-1).long().unsqueeze(dim=0)  #[1,H,W]
        img_left=torch.from_numpy(img_left).float().permute(2,0,1)  #[3,H,W]
        img_right=torch.from_numpy(img_right).float().permute(2,0,1)  #[3,H,W] 
        
        img_center=Resize(size=(384,512))(img_center)   #[3,h,w]
        label=Resize(size=(384,512),interpolation=InterpolationMode.NEAREST)(label).squeeze(dim=0)  #[h,w]
        img_left=Resize(size=(384,512))(img_left)   #[3,h,w]
        img_right=Resize(size=(384,512))(img_right)   #[3,h,w]
        
        x_dist=torch.tensor([col_l-5,col_r-5])
        y_dist=torch.tensor([row-5,row-5])
        
        return img_center,label,img_left,img_right,x_dist,y_dist
    
    def __len__(self):
        return len(self.img_path)


class SegUnlabelData(Dataset):
    def __init__(self,img_path):
        super(SegUnlabelData,self).__init__()
        self.img_path=img_path
    
    def __getitem__(self,index):
        path=self.img_path[index]
        row=np.random.choice(np.arange(1,10))
        col=np.random.choice(np.arange(1,10))
        img=np.array(Image.open(os.path.join(path,str(row)+'_'+str(col)+'.png')))/255  #[H,W,3]

        img=torch.from_numpy(img).float().permute(2,0,1)   #[3,H,W] 
        img=Resize(size=(384,512))(img)   #[9,3,h,w]
        
        return img
    
    def __len__(self):
        return len(self.img_path)


class SegUnlabelDatav2(Dataset):
    def __init__(self,img_path):
        super(SegUnlabelDatav2,self).__init__()
        self.img_path=img_path
    
    def __getitem__(self,index):
        path=self.img_path[index]
        row=np.random.choice(np.arange(1,10))
        col_l=np.random.choice(np.arange(1,6))
        col_r=col_l+4
        img_left=np.array(Image.open(os.path.join(path,str(row)+'_'+str(col_l)+'.png')))/255  #[H,W,3]
        img_right=np.array(Image.open(os.path.join(path,str(row)+'_'+str(col_r)+'.png')))/255  #[H,W,3]
        
        img_left=torch.from_numpy(img_left).float().permute(2,0,1)   #[3,H,W] 
        img_left=Resize(size=(384,512))(img_left)   #[3,h,w]

        img_right=torch.from_numpy(img_right).float().permute(2,0,1)   #[3,H,W] 
        img_right=Resize(size=(384,512))(img_right)   #[3,h,w]
        
        return img_left, img_right
    
    def __len__(self):
        return len(self.img_path)
    
    
class SegUnlabelDatav3(Dataset):
    def __init__(self,img_path):
        super(SegUnlabelDatav3,self).__init__()
        self.img_path=img_path
        
    def __getitem__(self,index):
        path=self.img_path[index]
        row=np.random.choice(np.arange(1,10))
        col_l=np.random.choice(np.arange(1,6))
        col_r=col_l+4
        img_left=np.array(Image.open(os.path.join(path,str(row)+'_'+str(col_l)+'.png')))/255  #[H,W,3]
        img_right=np.array(Image.open(os.path.join(path,str(row)+'_'+str(col_r)+'.png')))/255  #[H,W,3]
        
        img_left=torch.from_numpy(img_left).float().permute(2,0,1)   #[3,H,W] 
        img_left=Resize(size=(384,512))(img_left)   #[3,h,w]

        img_right=torch.from_numpy(img_right).float().permute(2,0,1)   #[3,H,W] 
        img_right=Resize(size=(384,512))(img_right)   #[3,h,w]
        
        row_any=np.random.choice(np.arange(1,10))
        col_any=np.random.choice(np.arange(1,10))
        while (row_any==row and col_any==col_l) or (row_any==row and col_any==col_r):
            row_any=np.random.choice(np.arange(1,10))
            col_any=np.random.choice(np.arange(1,10))
        img_any=np.array(Image.open(os.path.join(path,str(row_any)+'_'+str(col_any)+'.png')))/255  #[H,W,3]
        
        img_any=torch.from_numpy(img_any).float().permute(2,0,1)   #[3,H,W] 
        img_any=Resize(size=(384,512))(img_any)   #[3,h,w]
        
        x_dist=torch.tensor([col_l-col_any,col_r-col_any])
        y_dist=torch.tensor([row-row_any,row-row_any])
        
        return img_left, img_right, img_any, x_dist, y_dist
    
    def __len__(self):
        return len(self.img_path)
    

class SegValData(Dataset):
    def __init__(self,data_root):
        super(SegValData,self).__init__()
        self.img_path=[]
        for item in os.listdir(data_root):
            if not item.startswith('.'):
                self.img_path.append(os.path.join(data_root,item))
        
    def __getitem__(self,index):
        path=self.img_path[index]
        img=np.array(Image.open(os.path.join(path,'5_5.png')))/255  #[H,W,3]
            
        if path.split('/')[-3]=='Syn':
            label=np.load(os.path.join(path,'5_5_label.npy'))  #[H,W]
        else:
            label=np.load(os.path.join(path,'label.npy'))  #[H,W]
            
        img=torch.from_numpy(img).float().permute(2,0,1)  #[3,H,W]
        label=torch.from_numpy(label-1).long().unsqueeze(dim=0)  #[1,H,W]
        
        img=Resize(size=(384,512))(img)   #[3,h,w]
        label=Resize(size=(384,512),interpolation=InterpolationMode.NEAREST)(label).squeeze(dim=0)  #[h,w]
            
        return img,label
            
    def __len__(self):
        return len(self.img_path)


#Classification dataset
class ClassData(Dataset):
    def __init__(self,data_root):
        super(ClassData,self).__init__()
        self.img_path=[]
        for item in os.listdir(data_root):
            if not item.startswith('.'):
                self.img_path.append(os.path.join(data_root,item))
                
    def __getitem__(self,index):
        path=self.img_path[index]
        img=np.array(Image.open(os.path.join(path,'5_5.png')))/255  #[H,W,3]
        if path.split('/')[-3]=='Real':
            label=np.load(os.path.join(path,'label.npy'))  #[H,W] 
        else:
            label=np.load(os.path.join(path,'5_5_label.npy'))  #[H,W]
            
        img=torch.from_numpy(img).float().permute(2,0,1)  #[3,H,W]
        label=torch.from_numpy(label-1).long().unsqueeze(dim=0)  #[1,H,W]
        img,label=WeakAugment(img,label)
        img=StrongAugment(img.unsqueeze(dim=0)).squeeze(dim=0)  #[3,H,W]
        
        img=Resize(size=(384,512))(img)   #[3,h,w]
        label=Resize(size=(384,512),interpolation=InterpolationMode.NEAREST)(label).squeeze(dim=0)  #[h,w]
        
        num_classes=14
        label_onehot=torch.nn.functional.one_hot(label,num_classes)  #[h,w,14]
        label_onehot=torch.permute(label_onehot,dims=(2,0,1))  #[14,h,w]
        
        smooth=torch.rand((1,label.shape[0],label.shape[1]))*0.1  #[1,h,w]
        label_smooth = (1.0 - smooth) * label_onehot + smooth / num_classes  #[14,h,w]
        
        labels_class=[]
        inp_list=[]
        for i in range(num_classes):
            mask=label_smooth[i:i+1]  #[1,h,w]
            if torch.max(mask)>=0.9:
                inp_list.append(mask*img)  #[3,h,w]
                labels_class.append(i)
                
        inps=torch.stack(inp_list)  #[n,3,h,w]
        labels_class=torch.tensor(labels_class)  #[n]

        m=inps.shape[0]
        idx=torch.randperm(m)
        inps=inps[idx][:2]
        labels_class=labels_class[idx][:2]
        
        return inps,labels_class 

    def __len__(self):
        return len(self.img_path)
            