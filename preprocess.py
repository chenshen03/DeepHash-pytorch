import torch
from PIL import Image
from torchvision import transforms
#

## https://github.com/kuangliu/pytorch-cifar/issues/8
cifar_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                       std=[0.2023, 0.1994, 0.2010])
                                      #std=[0.2470, 0.2435, 0.2616])

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def image_train(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def image_test_10crop(resize_size=256, crop_size=224):
    return transforms.Compose([
         transforms.Resize(resize_size),
        transforms.TenCrop(crop_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])
