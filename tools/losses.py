import os 
import numpy as np
import scipy.ndimage as ndi
import torch
import matplotlib.pyplot as plt

def hd(out, gt, eps = 10e-5):

    # Compute distance for all pixels outside of the tumor
    d_xy_p0 = ndi.distance_transform_edt(gt==0)
    d_xy_p0[d_xy_p0>20] = np.max(d_xy_p0[d_xy_p0<20])

    # Compute distance for all pixels inside of the tumor
    d_xy_n0 = ndi.distance_transform_edt(gt!=0)
    
    # Normalizing the HD's
    d_max_p =torch.where(out!=0,torch.from_numpy(d_xy_p0).double(),torch.zeros(out.size()).double()).max()+eps #normalizing by maximum HD where px is not zero
    d_xy_p = d_xy_p0/d_max_p
    d_max_n = torch.where((1-out)!=0,torch.from_numpy(d_xy_n0).double(),torch.zeros(out.size()).double()).max()+eps#normalizing by maximum HD where 1-px is not zero
    d_xy_n = d_xy_n0/d_max_n

    # Loss terms
    first_term = out*d_xy_p.float() # Outside the tumor
    second_term = (1-out)*d_xy_n.float() # Inside the tumor

    loss1 = torch.sum(first_term)/((np.max([torch.sum(out[gt==0]),eps])))
    loss2 = torch.sum(second_term)/((np.max([torch.sum(1-out[gt!=0]),eps])))

    return loss1+loss2


def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft[:,1,...] - gt.float()) ** 2
    s_dtm = seg_dtm[:,1,...] ** 2
    g_dtm = gt_dtm[:,1,...] ** 2
    dtm = s_dtm + g_dtm
    multipled = torch.einsum('bxyz, bxyz->bxyz', delta_s, dtm)
    hd_loss = multipled.mean()

    return hd_loss

out=torch.zeros((120,120), requires_grad=True)
out2=torch.zeros((120,120),requires_grad=True)

gt = torch.zeros((120,120), requires_grad=False)

out[40:80,30:60]=1
out2[2:118,2:118]=1

gt[50:80,20:60]=1

hd_tensor = hd(out,gt)

hd(gt,gt)
hd(torch.ones_like(out),gt)
hd(torch.zeros_like(out),gt)

