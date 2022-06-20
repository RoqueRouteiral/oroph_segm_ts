import os
import glob
import torch
import random
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from scipy import ndimage
from tools.transforms import *
import elasticdeform


def get_loaders_xnet(cf, phase='train'):
    #patients = os.listdir(cf.dataset_path)
    patients_train = os.listdir(cf['dataset_path']+'train/')
    patients_val = os.listdir(cf['dataset_path']+'validation/')
    patients_extra_val = os.listdir(cf['dataset_path']+'extra_val/')
    patients_test = os.listdir(cf['dataset_path']+'test/')
    
    #whether to shuffle or not the images
    if cf['shuffle_data']:
        random.shuffle(patients_train)
        random.shuffle(patients_val)
        random.shuffle(patients_test)
        random.shuffle(patients_extra_val) 
###    #Creating Data Generator per split
    train_set = HeadyNeckDataset(patients_train, cf, 'train')
    val_set = HeadyNeckDataset(patients_val, cf, 'validation')
    test_set = HeadyNeckDataset(patients_test, cf, 'test')
    extra_val_set = HeadyNeckDataset(patients_extra_val, cf, 'extra_val')

    train_gen = DataLoader(train_set, batch_size=cf['batch_size'])
    val_gen = DataLoader(val_set, batch_size=cf['batch_size'])
    test_gen = DataLoader(test_set, batch_size=cf['batch_size'])
    extra_val_gen = DataLoader(extra_val_set, batch_size=cf['batch_size'])
    
    return train_gen, val_gen, test_gen, extra_val_gen


class HeadyNeckDataset(Dataset):
    """Heand and Neck Cancer dataset."""

    def __init__(self, indices, cf, phase):
        """
        Args:
            indices : list of the indices for this generator
            cf (Config file): set up info
            phase: train loader or eval loader. Important to apply or not DA.
        """
        self.indices = indices
        self.cf = cf
        self.phase = phase
        #self.transform = None
        self.transform = (self.cf['da_flip'] or 
                          self.cf['da_rot'] or  
                          self.cf['da_deform'])
        self.folder = self.phase + '/'

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx): 
        img_name_tc = os.path.join(self.cf['dataset_path'], self.folder, self.indices[idx], 'T1c.npy')
        img_name_t2 = os.path.join(self.cf['dataset_path'], self.folder, self.indices[idx], 'T2.npy')
        img_name_t1 = os.path.join(self.cf['dataset_path'], self.folder, self.indices[idx], 'T1.npy')   
        img_name_dw = os.path.join(self.cf['dataset_path'], self.folder, self.indices[idx], 'DWI.npy')   

        gt_name = os.path.join(self.cf['dataset_path'], self.folder, self.indices[idx], 'GT.npy')
        
        t1c_np = np.load(img_name_tc)       
        t2_np = np.load(img_name_t2)       
        t1_np = np.load(img_name_t1)   
        dw_np = np.load(img_name_dw)   
        
        # t1c_np=np.zeros_like(t1c_np)
        t2_np=np.zeros_like(t1c_np)
        t1_np=np.zeros_like(t1c_np)
        # dw_np=np.zeros_like(t1c_np)

        gt_np = np.load(gt_name)>0 #check this  
        gt_box=np.zeros_like(gt_np)
        coor = np.where(gt_np)
        gt_border= [np.min(coor[0]),np.max(coor[0])+1,np.min(coor[1]),np.max(coor[1])+1,np.min(coor[2]),np.max(coor[2])+1]
        gt_box[gt_border[0]:gt_border[1],gt_border[2]:gt_border[3],gt_border[4]:gt_border[5]]=1

        if self.cf['resize']:
            t1c_np = resize(t1c_np,(self.cf['sizeX'],self.cf['sizeY'],self.cf['sizeZ']),mode='constant')
            t2_np = resize(t2_np,(self.cf['sizeX'],self.cf['sizeY'],self.cf['sizeZ']),mode='constant')
            t1_np = resize(t1_np,(self.cf['sizeX'],self.cf['sizeY'],self.cf['sizeZ']),mode='constant')   
            dw_np = resize(dw_np,(self.cf['sizeX'],self.cf['sizeY'],self.cf['sizeZ']),mode='constant')            
            gt_np = resize(gt_np,(self.cf['sizeX'],self.cf['sizeY'],self.cf['sizeZ']),order=0,mode='constant')     
            gt_box = resize(gt_box,(self.cf['sizeX'],self.cf['sizeY'],self.cf['sizeZ']),order=0,mode='constant')     
        if self.cf['norm']:
            t1c_np = normalize(t1c_np)
#            t2_np = normalize(t2_np)
#            t1_np = normalize(t1_np)
            dw_np = normalize(dw_np)
                      
        #transforming them if da (custom transforms)
        if self.transform and self.phase == 'train': 
            t1c_np, dw_np, gt_np, gt_box = self.__transforms(t1c_np, dw_np, gt_np, gt_box)

           
        t1c = torch.from_numpy(t1c_np.copy()).unsqueeze(0).float()
        t2 = torch.from_numpy(t2_np.copy()).unsqueeze(0).float()
        t1 = torch.from_numpy(t1_np.copy()).unsqueeze(0).float()
        dw = torch.from_numpy(dw_np.copy()).unsqueeze(0).float()

        gt = torch.from_numpy(gt_np.copy().astype(float)).unsqueeze(0).float()
        gt_box = torch.from_numpy(gt_box.copy().astype(float)).unsqueeze(0).float()
#        print(gt.shape,gt_box.shape)
        images=torch.cat((t1c,t2,t1,dw),0)
        data = (images,gt,gt_box)
        patient_name = self.indices[idx]
        return data, patient_name
        
    def __transforms(self, im, imd, gt, gt_b):
        
        ##Image level
        if self.cf['da_flip'] and np.random.random() < 0.5:
            im, imd, gt, gt_b = vertical_flip3(im, imd, gt, gt_b)  
        if self.cf['da_rot']:    
            im, imd, gt, gt_b = random_rotation3( im, imd, gt, gt_b, self.cf['da_rot'])        
        if self.cf['da_deform'] and np.random.random() < 0.5:
             im, imd, gt, gt_b = elasticdeform.deform_random_grid([ im, imd, gt, gt_b], sigma=self.cf['da_deform'], points=3, order=[3,3,0,0])
        return im, imd, gt, gt_b