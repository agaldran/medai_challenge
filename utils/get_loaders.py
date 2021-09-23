import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from . import paired_transforms_tv04 as p_tr
import os, os.path as osp

class PolypDataset(Dataset):
    def __init__(self, csv_path, transforms=None, mean=None, std=None, test=False):
        self.csv_path=csv_path
        df = pd.read_csv(self.csv_path)
        self.im_list = df.image_path
        self.test=test
        if not self.test:
            self.target_list = df.mask_path
        else:
            self.target_list = None
        self.transforms = transforms
        self.normalize = tr.Normalize(mean, std)

    def __getitem__(self, index):
        # load image and labels
        im_name = self.im_list[index]
        img = Image.open(self.im_list[index])
        orig_size = img.size
        if not self.test:
            target = np.array(Image.open(self.target_list[index]).convert('L')) > 127
            target = Image.fromarray(target)
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            img = self.normalize(img)
            return img, target
        else:
            if self.transforms is not None:
                img = self.transforms(img)
            img = self.normalize(img)
            return img, im_name, orig_size

    def __len__(self):
        return len(self.im_list)

class InferenceDataset(Dataset):
    def __init__(self, data_source, transforms=None, mean=None, std=None):
        self.data_source = data_source
        self.target_list = None

        if self.data_source.endswith('.csv'):
            df = pd.read_csv(self.data_source)
            self.im_list = df.image_path
            if len(df.columns) == 2:
                self.target_list = df.mask_path
        else:
            self.im_list = os.listdir(self.data_source)

        self.transforms = transforms
        self.normalize = tr.Normalize(mean, std)

    def __getitem__(self, index):
        # load image and labels
        if self.data_source.endswith('.csv'):
            im_name = self.im_list[index]
            img = Image.open(im_name)
        else:
            im_name = osp.join(self.data_source, self.im_list[index])
            img = Image.open(im_name)
        orig_size = img.size
        if self.transforms is not None:
            if self.target_list is not None:
                target = np.array(Image.open(self.target_list[index]).convert('L')) > 127
                target = Image.fromarray(target)
                img, target = self.transforms(img, target)
                img = self.normalize(img)
                return img, target, im_name, orig_size
            else:
                img = self.transforms(img)
                img = self.normalize(img)
                return img, im_name, orig_size



    def __len__(self):
        return len(self.im_list)

def get_train_val_seg_datasets(csv_path_train, csv_path_val, mean=None, std=None, tg_size=(512, 512)):
    train_dataset = PolypDataset(csv_path=csv_path_train, mean=mean, std=std)
    val_dataset = PolypDataset(csv_path=csv_path_val, mean=mean, std=std)
    # transforms definition
    # required transforms
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    # geometric transforms
    h_flip = p_tr.RandomHorizontalFlip()
    v_flip = p_tr.RandomVerticalFlip()
    rotate = p_tr.RandomRotation(degrees=45, fill=0, fill_tg=0)

    scale = p_tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = p_tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = p_tr.RandomChoice([scale, transl, rotate])

    # intensity transforms
    brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
    jitter = p_tr.ColorJitter(brightness, contrast, saturation, hue)
    train_transforms = p_tr.Compose([resize, scale_transl_rot, jitter, h_flip, v_flip, tensorizer])
    val_transforms = p_tr.Compose([resize, tensorizer])
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms

    return train_dataset, val_dataset

def get_train_val_seg_loaders(csv_path_train, csv_path_val, batch_size=4, tg_size=(512, 512), mean=None, std=None,
                              num_workers=0):
    train_dataset, val_dataset = get_train_val_seg_datasets(csv_path_train, csv_path_val, tg_size=tg_size,
                                                            mean=mean, std=std)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader


def get_test_seg_dataset(csv_path_test, mean=None, std=None, tg_size=(512, 512)):
    # required transforms
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    test_transforms = tr.Compose([resize, tensorizer])

    test_dataset = PolypDataset(csv_path=csv_path_test, mean=mean, std=std, test=True)
    test_dataset.transforms = test_transforms

    return test_dataset

def get_test_seg_loader(csv_path_test, batch_size=4, tg_size=(512, 512), mean=None, std=None, num_workers=0):
    test_dataset = get_test_seg_dataset(csv_path_test, tg_size=tg_size, mean=mean, std=std)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return test_loader

def get_inference_seg_dataset(data_source, mean=None, std=None, tg_size=(512, 512)):
    # required transforms
    resize = p_tr.Resize(tg_size)
    tensorizer = p_tr.ToTensor()
    test_transforms = p_tr.Compose([resize, tensorizer])

    test_dataset = InferenceDataset(data_source=data_source, mean=mean, std=std)
    test_dataset.transforms = test_transforms

    return test_dataset

def get_inference_seg_loader(data_source, batch_size=4, tg_size=(512, 512), mean=None, std=None, num_workers=0):
    test_dataset = get_inference_seg_dataset(data_source, tg_size=tg_size, mean=mean, std=std)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return test_loader

