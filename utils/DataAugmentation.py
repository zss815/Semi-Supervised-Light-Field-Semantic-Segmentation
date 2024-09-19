from torchvision import transforms
import random


def StrongAugment(img):
    #img: [B,9,3,H,W]
    if random.random()<0.8:
        img=transforms.ColorJitter(brightness=[0.5,1], contrast=[0.5,1], saturation=[0.5,1], hue=0)(img)
    if random.random()<0.5:
        img=transforms.GaussianBlur(kernel_size=9, sigma=(0.5, 2.0))(img)
    if random.random()<0.2:
        img=transforms.Grayscale(num_output_channels=3)(img)
    return img


def WeakAugment(img,label):
    #img: [B,3,H,W] label: [B,H,W]
    if random.random()<0.5:
        img=transforms.functional.hflip(img)
        label=transforms.functional.hflip(label)
    return img, label