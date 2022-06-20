import os 
import numpy as np
import scipy.ndimage as ndi
import warnings
import yaml
import matplotlib.pyplot as plt
import logging
from torch.nn import Module
import shutil
import pickle
from skimage.measure import regionprops, label
from skimage.transform import resize
# from tools.metrics_dm import dice_dm, hd_dm, compute_surface_distances_dm
import pandas as pd
import seaborn as sns
# from models.utils_retinanet.anchors import Anchors
# from models.utils_retinanet.utils import BBoxTransform, ClipBoxes
# from tools.loader_detection import get_loaders_dt
import torch

def load_cf(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cf : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cf = yaml.load(stream)
    # Copy config file
    if not os.path.exists(cf['experiments']+cf['exp_name']):
        os.makedirs(cf['experiments']+cf['exp_name'])   
        
    if not os.path.exists(cf['experiments']+cf['exp_name']+'/additional_files/'):
        os.makedirs(cf['experiments']+cf['exp_name']+'/additional_files/')           
    
    if cf['train']:
        shutil.copyfile('config.yaml', os.path.join(cf['experiments'],cf['exp_name'], "config.yaml"))   
    #To finish this. This would be a nice feature to ensure reproductibility
    # for subdir, dirs, files in os.walk('E:/hecktor/scripts_segmentation/'):
    #     #print(dirs)
    #     dirs[:] = [d for d in dirs if d != 'Experiments/']
    #     for file in files:
    #         filepath = subdir + os.sep + file
    #         if filepath.endswith(".py"):
    #             print(filepath)
    #             shutil.copyfile(filepath,os.path.join(cf['experiments'],cf['exp_name'],'additional_files/', file))   
    return cf


def set_logger(cf):
    log=logging.getLogger()
    log.setLevel(logging.DEBUG)
#    logging.basicConfig(filename=os.path.join(cf['experiments'],cf['exp_name'],'logfile.log'),level=logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(os.path.join(cf['experiments'],cf['exp_name'],'logfile.log'),maxBytes=(1048576*5), backupCount=7)
    log.addHandler(handler)
    log.info('Logger is created')
#    if (log.hasHandlers()):
#        log.handlers.clear()
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    log.info('Logger is initialized')


##visualizing output results in three-view thumbnails
def thumbnail(image_np, gt_np, output_np, name, save_folder):
    gt_edge = make_contours(gt_np)
    output_edge = make_contours(output_np)

    rgb=np.repeat(np.expand_dims(image_np,3),3,axis=3)
    rgb=rgb/rgb.max()
    rgb[gt_edge>0]=(1,0,0)
    rgb[output_edge>0]=(0,0,1)
    rgb[(output_edge.astype(bool) & gt_edge.astype(bool))>0]=(1,0,1)
    rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())
    if (gt_np>0).any():
        x, y, z = np.where(gt_np)
        mid_slice_x = x.min() + (x.max() - x.min())//2
        mid_slice_y = y.min() + (y.max() - y.min())//2 
        mid_slice_z = z.min() + (z.max() - z.min())//2 
    
        contours_x = rgb[mid_slice_x,:,:]
        contours_y = rgb[:,mid_slice_y,:]
        contours_z = rgb[:,:,mid_slice_z]

        plt.imsave(save_folder+'/{}_axl.png'.format(name), contours_x)
        plt.imsave(save_folder+'/{}_cor.png'.format(name), contours_y)
        plt.imsave(save_folder+'/{}_sag.png'.format(name), contours_z)
        
def calc_iou_npy3D(a, b): #3D
    vol = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]) 
    #print((a,b))
    iw = np.min((a[3], b[3])) - np.max((a[0], b[0]))
    ih = np.min((a[4], b[4])) - np.max((a[1], b[1]))
    idp = np.min((a[5], b[5])) - np.max((a[2], b[2]))

    iw = np.clip(iw, a_min=0, a_max=None)
    ih = np.clip(ih, a_min=0, a_max=None)
    idp = np.clip(idp, a_min=0, a_max=None)

    ua = (a[3] - a[0]) * (a[4] - a[1]) * (a[5] - a[2]) + vol - iw * ih * idp

    ua = np.clip(ua, a_min=1e-8, a_max=None)

    intersection = iw * ih * idp

    IoU = intersection / ua

    return IoU

def calc_iou_npy(a, b): #3D
    vol = (b[2] - b[0]) * (b[3] - b[1]) 
    #print((a,b))
    iw = np.min((a[2], b[2])) - np.max((a[0], b[0]))
    ih = np.min((a[3], b[3])) - np.max((a[1], b[1]))

    iw = np.clip(iw, a_min=0, a_max=None)
    ih = np.clip(ih, a_min=0, a_max=None)

    ua = (a[2] - a[0]) * (a[3] - a[1]) + vol - iw * ih 

    ua = np.clip(ua, a_min=1e-8, a_max=None)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def clip_image(boxes,img_shape):
    height, width, depth = img_shape

    boxes[0] = np.clip(boxes[0], a_min=0, a_max=height)
    boxes[1] = np.clip(boxes[1], a_min=0, a_max=width)
    boxes[2] = np.clip(boxes[2], a_min=0, a_max=depth)


    boxes[3] = np.clip(boxes[3], a_min=0, a_max=height) #????? why was this inverted
    boxes[4] = np.clip(boxes[4], a_min=0, a_max=width)
    boxes[5] = np.clip(boxes[5], a_min=0, a_max=depth)
    return boxes

def overlap_box_and_image(image_np, box, name, save_folder):
    """
    Overlays the input image (with original resolution) and a given box. 
    Useful for debugging of the anchors and also visualization purposes.
    Note: the expected box is (x0,y0,z0,x1,y1,z1)
    Parameters
    ----------
    image_np : 3D numpy array
    input image in the original size (or resized as defined in the config.yaml)    
    bbox : 6 lentgh numpy array in the (x0,y0,z0,x1,y1,z1)
        DESCRIPTION.
    name : TYPE
        name of the input image for saving purposes.
    save_folder : TYPE
        path to save folder.

    Returns
    -------
    None.

    """
    # print(box)
    box = clip_image(box,image_np.shape)
    # print(box)

    mask = get_mask_from_coor_coco(box.astype(int),image_np.shape)
    mask_edge = make_contours(mask)
    print(mask.max())
    rgb=np.repeat(np.expand_dims(image_np,3),3,axis=3)
    rgb=rgb/rgb.max()
    rgb[mask_edge>0]=(1,0,0)
    rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())
    if (mask>0).any():
        x, y, z = np.where(mask)
        mid_slice_x = x.min() + (x.max() - x.min())//2
        mid_slice_y = y.min() + (y.max() - y.min())//2 
        mid_slice_z = z.min() + (z.max() - z.min())//2 
    
        contours_x = rgb[mid_slice_x,:,:]
        contours_y = rgb[:,mid_slice_y,:]
        contours_z = rgb[:,:,mid_slice_z]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.imsave(save_folder+'/{}_axl.png'.format(name), contours_x)
        plt.imsave(save_folder+'/{}_cor.png'.format(name), contours_y)
        plt.imsave(save_folder+'/{}_sag.png'.format(name), contours_z)
def overlap_all_anchors_and_image(image_np, all_anchors, name, save_folder):
    """
    Overlays the input image (with original resolution) and a given box. 
    Useful for debugging of the anchors and also visualization purposes.
    Note: the expected box is (x0,y0,z0,x1,y1,z1)
    Parameters
    ----------
    image_np : 3D numpy array
    input image in the original size (or resized as defined in the config.yaml)    
    list_of_anchors : 1xnr_anchx6 lentgh numpy array in the (x0,y0,z0,x1,y1,z1)
        DESCRIPTION.
    name : TYPE
        name of the input image for saving purposes.
    save_folder : TYPE
        path to save folder.

    Returns
    -------
    None.

    """
    mask = overlap_all_anchors(image_np,all_anchors)
    print(mask.max())
    rgb=np.repeat(np.expand_dims(image_np,3),3,axis=3)
    rgb=rgb/rgb.max()
    rgb[mask>0]=(1,0,0)
    rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())
    # if (mask>0).any():
    #     x, y, z = np.where(mask)
    #     mid_slice_x = x.min() + (x.max() - x.min())//2
    #     mid_slice_y = y.min() + (y.max() - y.min())//2 
    #     mid_slice_z = z.min() + (z.max() - z.min())//2 
    if (mask>0).any():
        mid_slice_x = 160//2
        mid_slice_y = 160//2 
        mid_slice_z = 160//2 
        contours_x = rgb[mid_slice_x,:,:]
        contours_y = rgb[:,mid_slice_y,:]
        contours_z = rgb[:,:,mid_slice_z]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.imsave(save_folder+'/{}_axl.png'.format(name), contours_x)
        plt.imsave(save_folder+'/{}_cor.png'.format(name), contours_y)
        plt.imsave(save_folder+'/{}_sag.png'.format(name), contours_z)
        
def overlap_all_anchors_and_image_oaat(image_np, all_anchors, name, save_folder,thr=0.999):
    """
    Overlays the input image (with original resolution) and a given box. 
    Useful for debugging of the anchors and also visualization purposes.
    Note: the expected box is (x0,y0,z0,x1,y1,z1)
    OOAT stands for one at a time, as its better to get the anchor and then 
    Parameters
    ----------
    image_np : 3D numpy array
    input image in the original size (or resized as defined in the config.yaml)    
    list_of_anchors : 1xnr_anchx6 lentgh numpy array in the (x0,y0,z0,x1,y1,z1)
        DESCRIPTION.
    name : TYPE
        name of the input image for saving purposes.
    save_folder : TYPE
        path to save folder.

    Returns
    -------
    None.

    """
    rgb=np.repeat(np.expand_dims(image_np,3),3,axis=3)
    rgb=rgb/rgb.max()
    for anch in range(all_anchors.shape[1]):
        # print(all_anchors[0,anch])
        if (all_anchors[0,anch][0]<120  and all_anchors[0,anch][3]>150): 
            # print(all_anchors[0,anch])
            continue
        elif np.random.rand(1)<thr:
            continue
        # elif 80 in all_anchors[0,anch]:
        #     continue
        box = all_anchors[0,anch].astype(int)
        mask = get_mask_from_coor_coco(box.astype(int),image_np.shape)
        mask_edge = make_contours(mask)
        print(mask.max())
    
        rgb[mask_edge>0]=(1,0,0)
        rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())
        if (mask>0).any():
            mid_slice_x = 250//2
            mid_slice_y = 250//2 
            mid_slice_z = 250//2 
            contours_x = rgb[mid_slice_x,:,:]
            contours_y = rgb[:,mid_slice_y,:]
            contours_z = rgb[:,:,mid_slice_z]
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            plt.imsave(save_folder+'/{}_axl.png'.format(name), contours_x)
            plt.imsave(save_folder+'/{}_cor.png'.format(name), contours_y)
            plt.imsave(save_folder+'/{}_sag.png'.format(name), contours_z)        
            
def overlap_all_anchors_and_image_with_gt(image_np, gt_np, all_anchors, name, save_folder,iou_thr=0.5):
    """
    Overlays the input image (with original resolution) and a given box. 
    Useful for debugging of the anchors and also visualization purposes.
    Note: the expected box is (x0,y0,z0,x1,y1,z1)
    OOAT stands for one at a time, as its better to get the anchor and then 
    Parameters
    ----------
    image_np : 3D numpy array
    input image in the original size (or resized as defined in the config.yaml)    
    list_of_anchors : 1xnr_anchx6 lentgh numpy array in the (x0,y0,z0,x1,y1,z1)
        DESCRIPTION.
    name : TYPE
        name of the input image for saving purposes.
    save_folder : TYPE
        path to save folder.

    Returns
    -------
    None.

    """
    rgb=np.repeat(np.expand_dims(image_np,3),3,axis=3)
    rgb=rgb/rgb.max()
    mid_slice_x = 250//2
    mid_slice_y = 250//2 
    mid_slice_z = 250//2 
    contours_x = rgb[mid_slice_x,:,:]
    contours_y = rgb[:,mid_slice_y,:]
    contours_z = rgb[:,:,mid_slice_z]
    gt_bb = gt_np
    mask = get_mask_from_coor_coco(gt_bb.astype(int),image_np.shape)
    mask_edge = make_contours(mask)
    rgb[mask_edge>0]=(0,1,0)
    rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.imsave(save_folder+'/{}_axl.png'.format(name), contours_x)
    plt.imsave(save_folder+'/{}_cor.png'.format(name), contours_y)
    plt.imsave(save_folder+'/{}_sag.png'.format(name), contours_z)   
    for anch in range(all_anchors.shape[1]):

        iou=calc_iou_npy(all_anchors[0,anch].astype(int),gt_bb[:6])
        # if iou>0.3:print(iou)
        if iou>iou_thr:# and all_anchors[0,anch][3]>140:
            print(iou)
            print(gt_bb[:6])
            print(all_anchors[0,anch].astype(int))        
            # if (iou<0.3 and iou>0.4 and all_anchors[0,anch][3]<140): 
            # print(all_anchors[0,anch])
            # continue
            box = all_anchors[0,anch].astype(int)
            mask = get_mask_from_coor_coco(box.astype(int),image_np.shape)
            mask_edge = make_contours(mask)
            print(mask.max())
        
            rgb[mask_edge>0]=(1,0,0)
            rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())
            if (mask>0).any():
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                contours_x = rgb[mid_slice_x,:,:]
                contours_y = rgb[:,mid_slice_y,:]
                contours_z = rgb[:,:,mid_slice_z]
                plt.imsave(save_folder+'/{}_axl.png'.format(name), contours_x)
                plt.imsave(save_folder+'/{}_cor.png'.format(name), contours_y)
                plt.imsave(save_folder+'/{}_sag.png'.format(name), contours_z)   
            
# img = inputs[0,0,]

# import pandas as pd

# annotations = pd.read_json('E:/project_2/images/debug/annotations_dwi.json')

# annot = annotations['HN001'].to_numpy()

# annot=annot_to_coco_fmt(annot) # changed this on 30-7 !
# annot_with_class = np.ones(len(annot)+1)
# annot_with_class[:6] = annot

# annot = resize_bbox_coor(annot_with_class, (160,160,160), (275,272,275))

def anchors_of_image(img,device):

    anchors = anchs(img,device)
    anchors = clipBoxes(anchors,img)
    return anchors


# import pprint

# from tools.misc import load_cf, set_logger
# cf = load_cf('config.yaml')
# pp = pprint.PrettyPrinter(indent=4)
# #Set the logger to write the logs of the training in train.log
# set_logger(cf)    
# device = torch.device("cuda:"+str(cf['gpu']) if torch.cuda.is_available() else "cpu")
# train_gen, _ = get_loaders_dt(cf)
# anchs = Anchors()
# clipBoxes = ClipBoxes()   

# traing = iter(train_gen)
# batch = next(traing)
# img, gt = batch
# anchors = anchors_of_image(img,device).detach().cpu().numpy()
# img=img[0,0,].cpu().detach().numpy()
# gt=gt[0,0,].cpu().detach().numpy()

def make_contours(img):
    img = img.astype(float) - ndi.binary_erosion(img, np.ones((3,3,3)))
    return img

def line_plots(metric, path_to_metrics, list_of_epochs, list_of_factors):
    """
    
    Parameters
    ----------
    metric : string
        'dices', 'hds' or 'msds'
    list_of_epochs : list
        [250, 300, 350, 400,450,500,550]    
    list_of_factors : list
        [1.3 ,1.25,1.2 , 1.15, 1.10, 1.05, 1.0]

    Returns
    -------
    None.

    """
    dices = []
    for ep in range(len(list_of_epochs)):
        this_dice = np.median(np.load(path_to_metrics + '/{}_{}_{}.npy'.format(metric,list_of_epochs[ep],list_of_factors[ep])))
        dices.append(this_dice)
    plt.plot(dices,list_of_epochs)

def get_mask_from_coor_coco(coor,img_shape):
    """

    Parameters
    ----------
    coor : np.array of length 6
        coordinates for the 3D boxes

    Returns
    -------
    binary mask of the bbox

    """
    mask = np.zeros(img_shape)
    # print(coor)
    mask[coor[0]:coor[2],coor[1]:coor[3]]=1
    return mask 
def visualize_output(img, gt, out):
    
    pass
            
def get_bboxes_metrics(path_to_images,scale=(160,160,160)):
    sizes = []
    ratios = []
    patients = os.listdir(path_to_images)
    annotations = pd.read_json('E:/project_2/images/debug/annotations_dwi.json')
    for i in range(3):
        gt = annotations[patients[i]].to_numpy()
        annot=annot_to_coco_fmt(gt) # changed this on 30-7 !
        annot_with_class = np.ones(len(annot)+1)
        annot_with_class[:6] = annot        
        resized_gt=resize_bbox_coor(annot_with_class, scale, (275,272,275))
        sizes.append(max(resized_gt[3]-resized_gt[0],resized_gt[4]-resized_gt[1],resized_gt[5]-resized_gt[2]))
        ratios.append(max(sizes[-1]/(resized_gt[3]-resized_gt[0]),sizes[-1]/(resized_gt[4]-resized_gt[1]),sizes[-1]/(resized_gt[5]-resized_gt[2])))
    
    return sizes, ratios

def overlap_all_anchors(img,all_anchors):
    img_all_anchors = np.zeros_like(img)
    for anch in range(all_anchors.shape[1]):
        coor = all_anchors[0,anch].astype(int)
        # print(coor)
        img_all_anchors[coor[0]:coor[3],coor[1]:coor[4],coor[2]:coor[5]]=+2
        img_all_anchors[coor[0]+1:coor[3]-1,coor[1]+1:coor[4]-1,coor[2]+1:coor[5]-1]=-1

    return img_all_anchors

def measure_shifts(path_to_predicted_out, save_images=False):
    patients = list(set([x[0:5] for x in os.listdir(path_to_predicted_out)]))
    shifts = np.ones((len(patients),6))*112
    pixel_spacing = [275/112* 0.79]
    for i in range(len(patients)):
        img = np.load(path_to_predicted_out + '{}_t1c_in.npy'.format(patients[i]))
        out = np.load(path_to_predicted_out + '{}_out.npy'.format(patients[i]))>0.5
        if not out.any():
            continue
        gt = np.load(path_to_predicted_out + '{}_true.npy'.format(patients[i]))
        coor_out = np.where(out)
        coor_gt = np.where(gt)
        out_border= [np.min(coor_out[0]),np.max(coor_out[0])+1,np.min(coor_out[1]),np.max(coor_out[1])+1,np.min(coor_out[2]),np.max(coor_out[2])+1]
        gt_border= [np.min(coor_gt[0]),np.max(coor_gt[0])+1,np.min(coor_gt[1]),np.max(coor_gt[1])+1,np.min(coor_gt[2]),np.max(coor_gt[2])+1]
    
        shifts[i]=np.round(pixel_spacing*(np.array(out_border)-np.array(gt_border)),1)
        if save_images:
            gt_bb = np.zeros_like(gt)
            box_bb = np.zeros_like(box)
            gt_bb[gt_border[0]:gt_border[1],gt_border[2]:gt_border[3],gt_border[4]:gt_border[5]]=1
            box_bb[box_border[0]:box_border[1],box_border[2]:box_border[3],box_border[4]:box_border[5]]=1
            affine = np.eye(4)
            affine[1,1] *= 0
            affine[2,2] *= 0  
            affine[1,2] = -1
            affine[2,1] = -1      
            img_nii = nib.Nifti1Image(img, affine)
            gt_nii = nib.Nifti1Image(gt_bb, affine)
            box_nii = nib.Nifti1Image(box_bb, affine)
    
    return shifts

def get_boxes_original_size(path_to_predicted_out, path_to_save_npy, path_to_thumb,saving=True,verbose=False):
    patients = list(set([x[0:5] for x in os.listdir(path_to_predicted_out)]))
    for i in range(len(patients)):
        img = np.load(path_to_predicted_out + '{}_t1c_in.npy'.format(patients[i]))
        out = np.load(path_to_predicted_out + '{}_out.npy'.format(patients[i]))>0.5
        if not out.any():
            out[0,0,0]=1 # this will eventually make it a nan and thus, remove it. As we say in the paper we did with the miss detection
        gt = np.load(path_to_predicted_out + '{}_true.npy'.format(patients[i]))>0.5
        coor_out = np.where(out)
        coor_gt = np.where(gt)

        box_border= [np.min(coor_out[0]),np.max(coor_out[0])+1,np.min(coor_out[1]),np.max(coor_out[1])+1,np.min(coor_out[2]),np.max(coor_out[2])+1]
        gt_border= [np.min(coor_gt[0]),np.max(coor_gt[0])+1,np.min(coor_gt[1]),np.max(coor_gt[1])+1,np.min(coor_gt[2]),np.max(coor_gt[2])+1]
        gt_bb = np.zeros_like(gt)
        box_bb = np.zeros_like(out)
        gt_bb[gt_border[0]:gt_border[1],gt_border[2]:gt_border[3],gt_border[4]:gt_border[5]]=1
        box_bb[box_border[0]:box_border[1],box_border[2]:box_border[3],box_border[4]:box_border[5]]=1
        out_big =resize(box_bb,(275,275,275))
        gt_big =resize(gt_bb,(275,275,275))    
        img_big =resize(img,(275,275,275))        
        if verbose:print(gt_big.min(),gt_big.max())
        if saving:
            if not os.path.exists(path_to_save_npy+'{}/'.format(patients[i])):
                os.makedirs(path_to_save_npy+'{}/'.format(patients[i]))
            if not os.path.exists(path_to_thumb):
                os.makedirs(path_to_thumb)
            thumbnail(img_big,gt_big,out_big,patients[i],path_to_thumb)
            np.save(path_to_save_npy+patients[i]+'/img.npy',img_big)
            np.save(path_to_save_npy+patients[i]+'/gt.npy',gt_big)
            np.save(path_to_save_npy+patients[i]+'/box.npy',out_big)

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"
        
class _Loss(Module):
    def __init__(self):
        super(_Loss, self).__init__()


class _WeightedLoss(_Loss):
    def __init__(self, weight=None):
        super(_WeightedLoss, self).__init__()
        self.register_buffer('weight', weight)
        
def dice(input, target, weight=None, smooth = 0.001):
    
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                      "Please ensure they have the same size.".format(target.size(), input.size()))
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    loss = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    return loss
              
class DiceLoss(_WeightedLoss):
    r"""Creates a criterion that measures the Dice Loss
    between the target and the output:

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If `reduce` is False, then `(N, *)`, same shape as
          input.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.DiceLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__(weight)

    def forward(self, input, target):
        _assert_no_grad(target)
        return 1-dice(input, target, weight=self.weight)
    
        
def resize_bbox_coor(annot, final_shape, original_shape):
    """
    Parameters
    ----------
    annot : np.array length 6
        Array of the GT annotations for this image.
    final_shape : tuple
        Factors to what the image is resized.
    original_shape: original size of the 3 axis of the image

    Returns
    -------
    res_annot : np.array length 6
        new GT annotations of the tumor

    """
    
    fx, fy = final_shape # final x, final y
    ox, oy = original_shape # original x, original y
    factors = (fx/ox, fy/oy)
    res_annot = np.zeros_like(annot)
    #print(factors)
    res_annot[0] = np.round(annot[0]*factors[0])
    res_annot[1] = np.round(annot[1]*factors[1])
    res_annot[2] = np.round(annot[2]*factors[0])
    res_annot[3] = np.round(annot[3]*factors[1])
    if (res_annot[0] != 0 and res_annot[1] !=0 and res_annot[2] !=0 and res_annot[3] !=0):
        res_annot[4] = 1 #annotation of class
    return res_annot#.astype(int)

def get_mask_from_coor(coor,img_shape):
    """

    Parameters
    ----------
    coor : np.array of length 6
        coordinates for the 3D boxes

    Returns
    -------
    binary mask of the bbox

    """
    mask = np.zeros(img_shape)
    mask[coor[0]:coor[1],coor[2]:coor[3]]=1
    return mask

def annot_to_coco_fmt(coor):
    """

    Parameters
    ----------
    coor : np.array of length 6
        x0,x1,y0,y1,

    Returns
    -------
    x0,y0,x1,y1
    
    Notes
    ------
    This can be easily done by a numpy function or even python indexing but i prefer it explicitly for clarity
    Important! COCOs format is actually different (xright,yright,w,h). But when loading the annotations the previous code immediately change it to this.

    """
    new_coor = np.zeros_like(coor)
    new_coor[0] = coor[0] 
    new_coor[1] = coor[2] 
    new_coor[2] = coor[1]    
    new_coor[3] = coor[3]
    
    return new_coor

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))    


def get_resulting_boxes(scores, labels, boxes, threshold = 0.5):
    results=[]
    
    if boxes.shape[0] > 0:
        # print(boxes.shape[0])
        # print(np.unique(boxes,axis=0).shape)
        # print(scores)
        # change to (x, y, z, w, h, d) 
        # boxes[:, 3] -= boxes[:, 0]
        # boxes[:, 4] -= boxes[:, 1]
        # boxes[:, 5] -= boxes[:, 2]
    
        # compute predicted labels and scores
        #for box, score, label in zip(boxes[0], scores[0], labels[0]):
        for box_id in range(boxes.shape[0]):
            score = float(scores[box_id])
            label = int(labels[box_id])
            box = boxes[box_id, :]
    
            # scores are sorted, so we can break
            if score < threshold:
                break
                
            # append detection to results
            results.append(box)

    return results

# def get_resulting_box(scores, labels, boxes, threshold = 0.5):
#     # print(scores)
#     if not np.array(scores).any(): return []
#     score = float(scores[0])
#     label = int(labels[0])
#     box = boxes[0, :]                
#     return box

def get_resulting_box_legacy(scores, labels, boxes, threshold = 0.5):
    # print(scores)
    if not np.array(scores).any(): return []
    score = float(scores[0])
    label = int(labels[0])
    box = boxes[0, :]                
    return box

def dice_box(box,labels,img_shape):
    mask_gt= get_mask_from_coor_coco(labels,img_shape)
    mask_out= get_mask_from_coor_coco(box,img_shape)
    return dice_dm(mask_gt,mask_out)

def hd_box(box,labels,img_shape,vox_spacing):
    mask_gt= get_mask_from_coor_coco(labels,img_shape)
    mask_out= get_mask_from_coor_coco(box,img_shape)
    surface_distances = compute_surface_distances_dm(mask_gt,mask_out,vox_spacing)
    return  hd_dm(surface_distances, 95)

def compute_metrics_detection_inference(nms_scores, nms_class, transformed_anchors,labels,img_shape,vox_spacing):
    bboxes=get_resulting_boxes(nms_scores,nms_class,transformed_anchors)
    dscs = []
    hds = []
    for box in bboxes:
        # print(box)
        # print(labels)
        dscs.append(dice_box(box.astype(int),labels.astype(int),img_shape))
        hds.append(hd_box(box.astype(int),labels.astype(int),img_shape,vox_spacing))
    dsc = np.mean(np.array(dscs))
    hd = np.mean(np.array(hds))
    return dsc, hd

def compute_metrics_detection(nms_scores, nms_class, transformed_anchors,labels):
    bboxes=get_resulting_boxes(nms_scores,nms_class,transformed_anchors)
    ious = []
    if not bboxes: return np.array([0])
    for box in bboxes:
        # print(box)
        # print(labels)
        ious.append(calc_iou_npy(box.astype(int),labels.astype(int)))
    iou = np.nanmean(np.array(ious))
    # print(iou)

    return iou

# def compute_metrics_detection_one_box(nms_scores, nms_class, transformed_anchors,labels):
#     batch_size = labels.shape[0]
#     ious = np.zeros((batch_size))
#     for bs in range(batch_size):
#         bbox=get_resulting_box(nms_scores,nms_class,transformed_anchors)
#         # print(bbox)
#         # print(labels)
#         # print(nms_scores)
#         if not np.array(bbox).any(): 
#             if np.array(labels[-1])==0:
#                 # print('no boxes')
#                 return np.array([1]) #(if the prediction and the box are both 0, perfect detection)
#             else: 
#                 return np.array([0])
#                 # print('no boxes and it should')
#         ious[bs] = calc_iou_npy(bbox.astype(int),labels[bs].astype(int))
#         print(iou)
#     return iou

def compute_metrics_detection_one_box(nms_scores, nms_class, transformed_anchors,labels):
    # bbox=get_resulting_box(nms_scores,nms_class,transformed_anchors)
    # print(bbox)
    # print(labels.shape)
    ious=np.zeros(labels.shape[0])
    for bs in range(labels.shape[0]):
        # print(nms_scores[bs][0])
        # print(labels[bs][0][-1])
        bbox=transformed_anchors[bs]
        # print(bbox)
        if (nms_scores[bs][0][0]<0.5): 
            # print('no boxes')
            # print(labels[bs][0][-1])
            if np.array(labels[bs][0][-1])==0:
                # print('no tumor')
                ious[bs]=np.array([1]) 
            else: 
                ious[bs]=np.array([0])
        else:
            # print(bbox[0],labels[bs][0])
            ious[bs] = calc_iou_npy(bbox[0].astype(int),labels[bs][0].astype(int))
    iou_mean=np.mean(ious)
    # print(iou_mean)
    return iou_mean

def predict_images(img, nms_scores, nms_class, transformed_anchors,labels,save_path,name):
    rgb=np.repeat(np.expand_dims(img,3),3,axis=2)
    rgb=rgb/rgb.max()
    gt_bb = labels
    mask = get_mask_from_coor_coco(labels.astype(int),img.shape)
    mask_edge = make_contours(mask)
    rgb[mask_edge>0]=(0,1,0)
    rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())    
    # print(transformed_anchors[0])
    if nms_scores[0]<0.5: 
        plt.imsave(save_path+'/{}.png'.format(name), rgb)
        return
    mask = get_mask_from_coor_coco(transformed_anchors[0].astype(int),img.shape)
    mask_edge = make_contours(mask)    
    rgb[mask_edge>0]=(1,0,0)
    rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())
    plt.imsave(save_path+'/{}.png'.format(name), rgb)
    return

def predict_images_npy(img, nms_scores, nms_class, transformed_anchors,labels,save_path_png,save_path_npy,name):
    bboxes=get_resulting_boxes(nms_scores,nms_class,transformed_anchors)
    rgb=np.repeat(np.expand_dims(img,3),3,axis=2)
    rgb=rgb/rgb.max()
    gt_bb = labels
    mask = get_mask_from_coor_coco(labels.astype(int),img.shape)
    mask_edge = make_contours(mask)
    rgb[mask_edge>0]=(0,1,0)
    rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())    
    for box in bboxes:
        mask = get_mask_from_coor_coco(box.astype(int),img.shape)
        mask_edge = make_contours(mask)    
        rgb[mask_edge>0]=(1,0,0)
        rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())
        if len(bboxes)>1: break
    np.save(save_path_npy+'/{}.npy'.format(name), mask)
    plt.imsave(save_path_png+'/{}.png'.format(name), rgb)
    return

def predict_all(imgs, nms_scores, nms_class, transformed_anchors,labels,save_path_png,save_path_npy,name):
    img=imgs[0,]
    # print(img.shape)
    rgb=np.repeat(np.expand_dims(img,3),3,axis=2)
    rgb=rgb/rgb.max()
    mask_gt = get_mask_from_coor_coco(labels.astype(int),img.shape)
    mask_edge_gt = make_contours(mask_gt)
    rgb[mask_edge_gt>0]=(0,1,0)
    # rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())  
    # print(transformed_anchors[0][0]==0)
    # print(transformed_anchors[0][1]==0)
    # print(transformed_anchors[0][2]<50)
    # print(transformed_anchors[0][3]<50)
    # print()
    if nms_scores[0]<0.5:
        
        # print('Saving a 0 prediction')
        mask = np.zeros_like(img)
    else:
        mask = get_mask_from_coor_coco(transformed_anchors[0].astype(int),img.shape)
        # print(transformed_anchors[0])
        mask_edge = make_contours(mask)    
        rgb[mask_edge>0]=(1,0,0)
        # rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())
    np.save(save_path_npy+'/{}_out.npy'.format(name), mask)
    np.save(save_path_npy+'/{}_true.npy'.format(name), mask_gt)
    np.save(save_path_npy+'/{}_t1c.npy'.format(name), imgs[0,])
    np.save(save_path_npy+'/{}_t2.npy'.format(name), imgs[1,])
    np.save(save_path_npy+'/{}_t1.npy'.format(name), imgs[2,])
    np.save(save_path_npy+'/{}_dwi.npy'.format(name), imgs[3,])    
    plt.imsave(save_path_png+'/{}.png'.format(name), rgb)
    return
def predict_all_old(imgs, nms_scores, nms_class, transformed_anchors,labels,save_path_png,save_path_npy,name):
    """Legacy"""
    img=imgs[0,]
    bboxes=get_resulting_boxes(nms_scores,nms_class,transformed_anchors)
    rgb=np.repeat(np.expand_dims(img,3),3,axis=2)
    rgb=rgb/rgb.max()
    mask_gt = get_mask_from_coor_coco(labels.astype(int),img.shape)
    mask_edge_gt = make_contours(mask_gt)
    rgb[mask_edge_gt>0]=(0,1,0)
    rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())  
    if not np.array(bboxes).any(): mask = np.zeros_like(img)
    for box in bboxes:
        mask = get_mask_from_coor_coco(box.astype(int),img.shape)
        mask_edge = make_contours(mask)    
        rgb[mask_edge>0]=(1,0,0)
        rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())
        if len(bboxes)>1: break
    np.save(save_path_npy+'/{}_out.npy'.format(name), mask)
    np.save(save_path_npy+'/{}_true.npy'.format(name), mask_gt)
    np.save(save_path_npy+'/{}_t1c.npy'.format(name), imgs[0,])
    np.save(save_path_npy+'/{}_t2.npy'.format(name), imgs[1,])
    np.save(save_path_npy+'/{}_t1.npy'.format(name), imgs[2,])
    np.save(save_path_npy+'/{}_dwi.npy'.format(name), imgs[3,])    
    plt.imsave(save_path_png+'/{}.png'.format(name), rgb)
    return
def overlay_image_box(img, box, save_path,name):
    rgb=np.repeat(np.expand_dims(img,3),3,axis=2)
    rgb=rgb/rgb.max()

    mask = get_mask_from_coor_coco(box.astype(int),img.shape)
    mask_edge = make_contours(mask)    
    rgb[mask_edge>0]=(1,0,0)
    rgb=(rgb-rgb.min())/(rgb.max()-rgb.min())
    plt.imsave(save_path+'/{}.png'.format(name), rgb)
    return

def get_boxplots(experiments):
    """
    This function creates boxplots for given paths to .npy files for two metrics (Dice and HD)

    Parameters
    ----------
    paths_to_dices : list
        list of strings containing the .npy to load the dices of particular experiments.
    paths_to_hds : list
        list of strings containing the .npy to load the hds of particular experiments.

    Returns
    -------
    None.

    """

    test_patients = os.listdir('E:/hecktor/data_npy/validation/')
    path_to_metrics='E:/hecktor/scripts_segmentation/Experiments/'
    
    dice_ct=np.load(path_to_metrics+'{}/dices.npy'.format(experiments[0]))
    dice_pt=np.load(path_to_metrics+'{}/dices.npy'.format(experiments[1]))
    dice_bm=np.load(path_to_metrics+'{}/dices.npy'.format(experiments[2]))
    
    hd_ct=np.load(path_to_metrics+'{}/hds.npy'.format(experiments[0]))
    hd_pt=np.load(path_to_metrics+'{}/hds.npy'.format(experiments[1]))
    hd_bm=np.load(path_to_metrics+'{}/hds.npy'.format(experiments[2]))
    
    
    labels=['CT', 'PET', 'Both']
    
    df_dice=pd.DataFrame(data=[dice_ct, dice_pt, dice_bm], index=labels, columns=test_patients)
    df_hd=pd.DataFrame(data=[hd_ct, hd_pt, hd_bm], index=labels, columns=test_patients)
    
    # Plotting the features using boxes
    sns.set_style("white")
    plt.rcParams.update({'font.size': 40})
    plt.subplot(1,2,1)
    axes = plt.gca()
    sns.boxplot(data = df_dice.T, width=0.4) 
    plt.ylabel('Dice')
    axes.set_ylim(0,1)
    
    plt.subplot(1,2,2)
    axes = plt.gca()
    sns.boxplot(data = df_hd.T, width=0.4) 
    plt.ylabel('95th HD (mm)')

    axes.set_ylim(0,60)
    
def get_boxplots_dwi(dice_t1c,dice_dwi,dice_b,hd_t1c,hd_dwi,hd_b,msd_t1c,msd_dwi,msd_b):
    """
    This function creates boxplots for given paths to .npy files for two metrics (Dice and HD)

    Parameters
    ----------
    paths_to_dices : list
        list of strings containing the .npy to load the dices of particular experiments.
    paths_to_hds : list
        list of strings containing the .npy to load the hds of particular experiments.

    Returns
    -------
    None.

    """
    
    labels=['T1Gd', 'DWI', 'Both']
    
    df_dice=pd.DataFrame(data=[dice_t1c, dice_dwi, dice_b], index=labels)
    df_hd=pd.DataFrame(data=[hd_t1c, hd_dwi, hd_b], index=labels)
    df_msd=pd.DataFrame(data=[msd_t1c, msd_dwi, msd_b], index=labels)
    # Plotting the features using boxes
    sns.set_style("white")
    plt.rcParams.update({'font.size': 40})
    plt.subplot(1,3,1)
    axes = plt.gca()
    sns.boxplot(data = df_dice.T, width=0.4) 
    plt.ylabel('Dice')
    axes.set_ylim(0,1)
    
    plt.subplot(1,3,2)
    axes = plt.gca()
    sns.boxplot(data = df_hd.T, width=0.4) 
    plt.ylabel('95th HD (mm)')

    axes.set_ylim(0,60)   

    plt.subplot(1,3,3)
    axes = plt.gca()
    sns.boxplot(data = df_msd.T, width=0.4) 
    plt.ylabel('MSD (mm)')

    axes.set_ylim(0,10)  
    
    
def get_boxplots_shifts(shifts_t1c,shifts_dwi,shifts_both):
    """
    This function creates boxplots for given paths to .npy files for two metrics (Dice and HD)

    Parameters
    ----------
    paths_to_dices : list
        list of strings containing the .npy to load the dices of particular experiments.
    paths_to_hds : list
        list of strings containing the .npy to load the hds of particular experiments.

    Returns
    -------
    None.

    """
    
    labels=['T1Gd', 'DWI', 'Both']
    
    df_dice=pd.DataFrame(data=[shifts_t1c, shifts_dwi, shifts_both], index=labels)

    # Plotting the features using boxes
    plt.figure()
    sns.set_style("white")
    plt.rcParams.update({'font.size': 40})
    axes = plt.gca()
    sns.boxplot(data = df_dice.T, width=0.4) 
    plt.ylabel('Mean shift in 6 directions (mm)')
    axes.set_ylim(0,30)
    
    
def get_boxplots_4(dice_1,dice_2,dice_3, dice_4,
                   hd_1,hd_2,hd_3,hd_4, 
                   msd_1,msd_2,msd_3,msd_4):
    """
    This function creates boxplots for given paths to .npy files for two metrics (Dice and HD)

    Parameters
    ----------
    paths_to_dices : list
        list of strings containing the .npy to load the dices of particular experiments.
    paths_to_hds : list
        list of strings containing the .npy to load the hds of particular experiments.

    Returns
    -------
    None.

    """
    
    labels=['UNet', 'YNet', 'XNet', 'Xfusion']
    
    df_dice=pd.DataFrame(data=[dice_1,dice_2,dice_3, dice_4], index=labels)
    df_hd=pd.DataFrame(data=[hd_1,hd_2,hd_3,hd_4], index=labels)
    df_msd=pd.DataFrame(data=[msd_1,msd_2,msd_3,msd_4], index=labels)
    # Plotting the features using boxes
    sns.set_style("white")
    plt.rcParams.update({'font.size': 25})
    plt.subplot(1,3,1)
    axes = plt.gca()
    sns.boxplot(data = df_dice.T, width=0.4) 
    plt.ylabel('Dice')
    axes.set_ylim(0,1)
    
    plt.subplot(1,3,2)
    axes = plt.gca()
    sns.boxplot(data = df_hd.T, width=0.4) 
    plt.ylabel('95th HD (mm)')

    axes.set_ylim(0,60)   

    plt.subplot(1,3,3)
    axes = plt.gca()
    sns.boxplot(data = df_msd.T, width=0.4) 
    plt.ylabel('MSD (mm)')

    axes.set_ylim(0,10)  
    
def get_boxplots_3(dice_1,dice_2,dice_3,
                   hd_1,hd_2,hd_3,
                   msd_1,msd_2,msd_3,
                   dice_ref=np.nan,hd_ref=np.nan,msd_ref=np.nan):
    """
    This function creates boxplots for given paths to .npy files for two metrics (Dice and HD)

    Parameters
    ----------
    paths_to_dices : list
        list of strings containing the .npy to load the dices of particular experiments.
    paths_to_hds : list
        list of strings containing the .npy to load the hds of particular experiments.

    Returns
    -------
    None.

    """
    
    labels=['AS', 'DWI', 'AS+DWI']
    
    df_dice=pd.DataFrame(data=[dice_1,dice_2,dice_3], index=labels)
    df_hd=pd.DataFrame(data=[hd_1,hd_2,hd_3], index=labels)
    df_msd=pd.DataFrame(data=[msd_1,msd_2,msd_3], index=labels)
    # Plotting the features using boxes
    sns.set_style("white")
    plt.rcParams.update({'font.size': 25})
    plt.subplot(1,3,1)
    axes = plt.gca()
    sns.boxplot(data = df_dice.T, width=0.4, linewidth=3, color="gray") 
    plt.axhline(y=np.median(dice_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('Dice')
    axes.set_ylim(0,1)
    
    plt.subplot(1,3,2)
    axes = plt.gca()
    sns.boxplot(data = df_hd.T, width=0.4, linewidth=3, color="gray") 
    plt.axhline(y=np.median(hd_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('95th HD (mm)')

    axes.set_ylim(0,60)   

    plt.subplot(1,3,3)
    axes = plt.gca()
    sns.boxplot(data = df_msd.T, width=0.4, linewidth=3, color="gray") 
    plt.axhline(y=np.median(msd_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('MSD (mm)')

    axes.set_ylim(0,10)   
    
def get_boxplots_fig_4(dice_1,dice_2,dice_3,dice_4,
                   hd_1,hd_2,hd_3,hd_4,
                   msd_1,msd_2,msd_3,msd_4,
                   dice_ref=np.nan,hd_ref=np.nan,msd_ref=np.nan):
    """
    This function creates boxplots for given paths to .npy files for two metrics (Dice and HD)

    Parameters
    ----------
    paths_to_dices : list
        list of strings containing the .npy to load the dices of particular experiments.
    paths_to_hds : list
        list of strings containing the .npy to load the hds of particular experiments.

    Returns
    -------
    None.

    """
    
    labels=['AS DET', 'DWI', 'ALL', 'AS SEG']
    
    df_dice=pd.DataFrame(data=[dice_1,dice_2,dice_3,dice_4], index=labels)
    df_hd=pd.DataFrame(data=[hd_1,hd_2,hd_3,hd_4], index=labels)
    df_msd=pd.DataFrame(data=[msd_1,msd_2,msd_3,msd_4], index=labels)
    # Plotting the features using boxes
    sns.set_style("white")
    custom = ["#717D7E", "#717D7E", "#717D7E", "#C1C1C1"]
    sns.set_palette(sns.color_palette(custom))
    plt.rcParams.update({'font.size': 20})
    plt.subplot(1,3,1)
    axes = plt.gca()
    sns.boxplot(data = df_dice.T, width=0.4, linewidth=3) 
    plt.axhline(y=np.median(dice_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('Dice')
    axes.set_ylim(0,1)
    
    plt.subplot(1,3,2)
    axes = plt.gca()
    sns.boxplot(data = df_hd.T, width=0.4, linewidth=3) 
    plt.axhline(y=np.median(hd_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('95th HD (mm)')

    axes.set_ylim(0,60)   

    plt.subplot(1,3,3)
    axes = plt.gca()
    sns.boxplot(data = df_msd.T, width=0.4, linewidth=3) 
    plt.axhline(y=np.median(msd_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('MSD (mm)')

    axes.set_ylim(0,10)   
    

    
def get_boxplots_fig_2_4boxplots(dice_1,dice_2,dice_3,dice_4,
                   hd_1,hd_2,hd_3,hd_4,
                   msd_1,msd_2,msd_3,msd_4,
                   dice_ref=np.nan,hd_ref=np.nan,msd_ref=np.nan,
                   labels=['AS', 'DWI', 'AS+DWI', 'UF Loss']):
    """
    This function creates boxplots for given paths to .npy files for two metrics (Dice and HD)

    Parameters
    ----------
    paths_to_dices : list
        list of strings containing the .npy to load the dices of particular experiments.
    paths_to_hds : list
        list of strings containing the .npy to load the hds of particular experiments.

    Returns
    -------
    None.

    """
    
    
    
    df_dice=pd.DataFrame(data=[dice_1,dice_2,dice_3,dice_4], index=labels)
    df_hd=pd.DataFrame(data=[hd_1,hd_2,hd_3,hd_4], index=labels)
    df_msd=pd.DataFrame(data=[msd_1,msd_2,msd_3,msd_4], index=labels)
    # Plotting the features using boxes
    sns.set_style("white")
    custom = ["#717D7E", "#717D7E", "#717D7E", "#717D7E"]
    sns.set_palette(sns.color_palette(custom))
    plt.rcParams.update({'font.size': 20})
    plt.subplot(1,3,1)
    axes = plt.gca()
    sns.boxplot(data = df_dice.T, width=0.4, linewidth=3) 
    plt.axhline(y=np.median(dice_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('Dice')
    axes.set_ylim(0,1)
    
    plt.subplot(1,3,2)
    axes = plt.gca()
    sns.boxplot(data = df_hd.T, width=0.4, linewidth=3) 
    plt.axhline(y=np.median(hd_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('95th HD (mm)')

    axes.set_ylim(0,60)   

    plt.subplot(1,3,3)
    axes = plt.gca()
    sns.boxplot(data = df_msd.T, width=0.4, linewidth=3) 
    plt.axhline(y=np.median(msd_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('MSD (mm)')

    axes.set_ylim(0,10) 
    
    
def diff_boxplots(dice_1, dice_2, dice_3, dice_ref,
                   hd_1,hd_2,hd_3,hd_ref, 
                   msd_1,msd_2,msd_3,msd_ref):
    """
    This function creates boxplots for given paths to .npy files for two metrics (Dice and HD)

    Parameters
    ----------
    paths_to_dices : list
        list of strings containing the .npy to load the dices of particular experiments.
    paths_to_hds : list
        list of strings containing the .npy to load the hds of particular experiments.

    Returns
    -------
    None.

    """
    
    labels=['YNet', 'XNet', 'ANet']
    
    df_dice=pd.DataFrame(data=[dice_1-dice_ref,dice_2-dice_ref,dice_3-dice_ref], index=labels)
    df_hd=pd.DataFrame(data=[hd_1-hd_ref,hd_2-hd_ref,hd_3-hd_ref], index=labels)
    df_msd=pd.DataFrame(data=[msd_1-msd_ref,msd_2-msd_ref,msd_3-msd_ref], index=labels)
    # Plotting the features using boxes
    sns.set_style("white")
    plt.rcParams.update({'font.size': 40})
    plt.subplot(1,3,1)
    axes = plt.gca()
    sns.boxplot(data = df_dice.T, width=0.4) 
    plt.ylabel('Dice')
    axes.set_ylim(-0.3,0.3)
    
    plt.subplot(1,3,2)
    axes = plt.gca()
    sns.boxplot(data = df_hd.T, width=0.4) 
    plt.ylabel('95th HD (mm)')

    axes.set_ylim(-30,30)   

    plt.subplot(1,3,3)
    axes = plt.gca()
    sns.boxplot(data = df_msd.T, width=0.4) 
    plt.ylabel('MSD (mm)')

    axes.set_ylim(-5,5)      

    
def diff_boxplots_one(dice_1, dice_ref,
                   hd_1,hd_ref, 
                   msd_1,msd_ref):
    """
    This function creates boxplots for given paths to .npy files for two metrics (Dice and HD)

    Parameters
    ----------
    paths_to_dices : list
        list of strings containing the .npy to load the dices of particular experiments.
    paths_to_hds : list
        list of strings containing the .npy to load the hds of particular experiments.

    Returns
    -------
    None.

    """
    
    # labels=['Dice', '95th HD (mm)', 'MSD (mm)']
    
    df_dice=pd.DataFrame(data=[dice_1-dice_ref])
    df_hd=pd.DataFrame(data=[hd_1-hd_ref])
    df_msd=pd.DataFrame(data=[msd_1-msd_ref])
    # Plotting the features using boxes
    sns.set_style("white")
    plt.rcParams.update({'font.size': 40})
    plt.subplot(1,3,1)
    axes = plt.gca()
    sns.boxplot(data = df_dice.T, width=0.4) 
    plt.ylabel('Dice')
    axes.set_ylim(-0.3,0.3)
    
    plt.subplot(1,3,2)
    axes = plt.gca()
    sns.boxplot(data = df_hd.T, width=0.4) 
    plt.ylabel('95th HD (mm)')

    axes.set_ylim(-30,30)   

    plt.subplot(1,3,3)
    axes = plt.gca()
    sns.boxplot(data = df_msd.T, width=0.4) 
    plt.ylabel('MSD (mm)')

    axes.set_ylim(-5,5)   
    
def get_boxplot(dice_1,
                   hd_1,
                   msd_1,
                   dice_ref=np.nan,hd_ref=np.nan,msd_ref=np.nan):
    """
    This function creates boxplots for given paths to .npy files for two metrics (Dice and HD)

    Parameters
    ----------
    paths_to_dices : list
        list of strings containing the .npy to load the dices of particular experiments.
    paths_to_hds : list
        list of strings containing the .npy to load the hds of particular experiments.

    Returns
    -------
    None.

    """
    
    
    df_dice=pd.DataFrame(data=[dice_1])
    df_hd=pd.DataFrame(data=[hd_1])
    df_msd=pd.DataFrame(data=[msd_1])
    # Plotting the features using boxes
    sns.set_style("white")
    plt.rcParams.update({'font.size': 25})
    plt.subplot(1,3,1)
    axes = plt.gca()
    sns.boxplot(data = df_dice.T, width=0.4, linewidth=3, color="gray") 
    plt.axhline(y=np.median(dice_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('Dice')
    axes.set_ylim(0,1)
    
    plt.subplot(1,3,2)
    axes = plt.gca()
    sns.boxplot(data = df_hd.T, width=0.4, linewidth=3, color="gray") 
    plt.axhline(y=np.median(hd_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('95th HD (mm)')

    axes.set_ylim(0,60)   

    plt.subplot(1,3,3)
    axes = plt.gca()
    sns.boxplot(data = df_msd.T, width=0.4, linewidth=3, color="gray") 
    plt.axhline(y=np.median(msd_ref), color='#ff3300', linestyle='--', linewidth=3)
    plt.ylabel('MSD (mm)')

    axes.set_ylim(0,10)   