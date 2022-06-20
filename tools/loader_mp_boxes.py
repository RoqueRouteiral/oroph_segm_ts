import os
import glob
import torch
import random
import logging
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from scipy import ndimage
from tools.transforms import *
import elasticdeform


def get_loaders_boxes(cf, phase='train'):
    #patients = os.listdir(cf.dataset_path)
    patients_train = os.listdir(cf['dataset_path']+'train/')
    patients_val = os.listdir(cf['dataset_path']+'validation/')
    patients_test = os.listdir(cf['dataset_path']+'test/')
    
    #whether to shuffle or not the images
    if cf['shuffle_data']:
        random.shuffle(patients_train)
        random.shuffle(patients_val)
        random.shuffle(patients_test)
###    #Creating Data Generator per split
    train_set = HeadyNeckDataset(patients_train, cf, 'train')
    val_set = HeadyNeckDataset(patients_val, cf, 'validation')
    test_set = HeadyNeckDataset(patients_test, cf, 'test')

    train_gen = DataLoader(train_set, batch_size=cf['batch_size'])
    val_gen = DataLoader(val_set, batch_size=cf['batch_size'])
    test_gen = DataLoader(test_set, batch_size=cf['batch_size'])
    
    return train_gen, val_gen, test_gen


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
        
        if not self.cf['t1gd']: t1c_np=np.zeros_like(t1c_np)
        if not self.cf['t2w']:  t2_np=np.zeros_like(t1c_np)
        if not self.cf['t1w']:  t1_np=np.zeros_like(t1c_np)
        if not self.cf['dwi']:  dw_np=np.zeros_like(t1c_np)

        gt_np = np.load(gt_name)>0 #check this  
        
        gt_dist = ndimage.distance_transform_edt(gt_np==0)
        gt_dist = gt_dist<1
        coor = np.where(gt_dist)
        if self.phase == 'train':
#            borders=np.random.randint(0, high=self.cf.max_shift, size=6)
            borders= 8 * np.random.randn(6) + 17 #For the toy version : np.ones(6).astype(int)*8
        else:
            borders=np.ones(6).astype(int)*8

            
        new_borders=np.array([np.min(coor[0])-borders[0],np.max(coor[0])+borders[1]+1,
                              np.min(coor[1])-borders[2],np.max(coor[1])+borders[3]+1,
                              np.min(coor[2])-borders[4],np.max(coor[2])+borders[5]+1])
        #print(new_borders)
        new_borders[new_borders<0]=0
        new_borders[1]=np.min((t1c_np.shape[0],new_borders[1]))
        new_borders[3]=np.min((t1c_np.shape[1],new_borders[3]))
        new_borders[5]=np.min((t1c_np.shape[2],new_borders[5]))        
            
        #Now we make sure that they are not more than 112 of width
        if new_borders[1]-new_borders[0]>112:new_borders[1]=new_borders[1]-(new_borders[1]-new_borders[0]-112)
        if new_borders[3]-new_borders[2]>112:new_borders[3]=new_borders[3]-(new_borders[3]-new_borders[2]-112)
        if new_borders[5]-new_borders[4]>112:new_borders[5]=new_borders[5]-(new_borders[5]-new_borders[4]-112)
        #print(new_borders)
        new_borders=new_borders.astype(int)
        cropped_t1c = t1c_np[new_borders[0]:new_borders[1], new_borders[2]:new_borders[3], new_borders[4]:new_borders[5]]
        cropped_t2 = t2_np[new_borders[0]:new_borders[1], new_borders[2]:new_borders[3], new_borders[4]:new_borders[5]]
        cropped_t1 = t1_np[new_borders[0]:new_borders[1], new_borders[2]:new_borders[3], new_borders[4]:new_borders[5]]
        cropped_dw = dw_np[new_borders[0]:new_borders[1], new_borders[2]:new_borders[3], new_borders[4]:new_borders[5]]
        cropped_gt = gt_np[new_borders[0]:new_borders[1], new_borders[2]:new_borders[3], new_borders[4]:new_borders[5]]

        cropped_t1c=np.pad(cropped_t1c,((0,112-(new_borders[1]-new_borders[0])),(0,112-(new_borders[3]-new_borders[2])),(0,112-(new_borders[5]-new_borders[4]))),mode='constant')
        cropped_t2=np.pad(cropped_t2,((0,112-(new_borders[1]-new_borders[0])),(0,112-(new_borders[3]-new_borders[2])),(0,112-(new_borders[5]-new_borders[4]))),mode='constant')
        cropped_t1=np.pad(cropped_t1,((0,112-(new_borders[1]-new_borders[0])),(0,112-(new_borders[3]-new_borders[2])),(0,112-(new_borders[5]-new_borders[4]))),mode='constant')
        cropped_dw=np.pad(cropped_dw,((0,112-(new_borders[1]-new_borders[0])),(0,112-(new_borders[3]-new_borders[2])),(0,112-(new_borders[5]-new_borders[4]))),mode='constant')

        cropped_gt=np.pad(cropped_gt,((0,112-(new_borders[1]-new_borders[0])),(0,112-(new_borders[3]-new_borders[2])),(0,112-(new_borders[5]-new_borders[4]))),mode='constant')        

        t1c_np=cropped_t1c
        t2_np=cropped_t2
        t1_np=cropped_t1
        dw_np=cropped_dw
        
        gt_np=cropped_gt   
        
        if self.cf['norm']:
            if self.cf['t1gd'] and t1c_np.any(): t1c_np = normalize(t1c_np)
            if self.cf['t2w'] and t2_np.any(): t2_np = normalize(t2_np)
            if self.cf['t1w'] and t1_np.any(): t1_np = normalize(t1_np)
            if self.cf['dwi'] and dw_np.any(): dw_np = normalize(dw_np)

#        logging.info((self.indices[idx],t1c_np.max(),t2_np.max(),t1_np.max(),dw_np.max()))
        t1c = torch.from_numpy(t1c_np.copy()).unsqueeze(0).float()
        t2 = torch.from_numpy(t2_np.copy()).unsqueeze(0).float()
        t1 = torch.from_numpy(t1_np.copy()).unsqueeze(0).float()
        dw = torch.from_numpy(dw_np.copy()).unsqueeze(0).float()

        gt = torch.from_numpy(gt_np.copy().astype(float)).unsqueeze(0).float()
        
        images=torch.cat((t1c,t2,t1,dw),0)
        data = (images,gt)
        patient_name = self.indices[idx]
        return data, patient_name
        
    