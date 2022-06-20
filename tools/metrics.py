import numpy as np
import warnings
from torch.nn import Module
import scipy.ndimage as ndi
import torch
from skimage.transform import resize
from scipy.ndimage.measurements import label
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

class GeneralizedDiceLoss(Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """
    
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, input, target):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        
        input = torch.flatten(input)
        target = torch.flatten(target)
        target = target.float()


#        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
        input = torch.unsqueeze(input,0)
        target = torch.unsqueeze(target,0)
        input = torch.cat((input, 1 - input), dim=0)
        target = torch.cat((target, 1 - target), dim=0)
        
        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
#        print(w_l)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False
        
        intersect = (input * target).sum(-1)
        intersect = intersect * w_l
        
        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)
#        print(1 - 2 * (intersect.sum() / denominator.sum())  )
        return 1 - 2 * (intersect.sum() / denominator.sum())  

################################
#     Asymmetric Focal loss    #
################################
# From: https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py and the paper entitled:
#  Unified Focal loss: Generalising Dice and
# cross entropy-based losses to handle class imbalanced
# medical image segmentation
# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')
    
def asymmetric_focal_loss(y_true, y_pred,delta=0.7, gamma=2.):

    # axis = identify_axis(y_true.shape)  

    epsilon = 1e-10
    y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * torch.log(y_pred)
    
    #calculate losses separately for each class, only suppressing background class
    # back_ce = torch.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
    back_ce = torch.pow(1 - y_pred, gamma) * (torch.ones_like(cross_entropy)-cross_entropy) #check
    back_ce =  (1 - delta) * back_ce

    fore_ce = cross_entropy
    fore_ce = delta * fore_ce

    loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce],axis=-1),axis=-1))
    print('FO_loss: ',loss)
    return loss


#################################
# Asymmetric Focal Tversky loss #
#################################

def asymmetric_focal_tversky_loss(y_true, y_pred, delta=0.7, gamma=0.75):
    # Clip values to prevent division by zero error
    epsilon = 1e-10
    y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)

    axis = identify_axis(y_true.shape)
    print('axis: ',axis)
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
    tp = torch.sum(y_true * y_pred, axis=axis)
    fn = torch.sum(y_true * (1-y_pred), axis=axis)
    fp = torch.sum((1-y_true) * y_pred, axis=axis)
    dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)
    print('Dice_TV: ', dice_class)
    print('Dice_TV,0: ', dice_class[:,0])
    print('Dice_TV,1: ', dice_class[:,1])

    #calculate losses separately for each class, only enhancing foreground class
    # back_dice = (1-dice_class[:,0]) 
    # fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -gamma) 
    back_dice = (1-dice_class[:,0]) 
    fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -gamma) 
    # Average class scores
    loss = torch.mean(torch.stack([back_dice,fore_dice],axis=-1))
    print('TV_loss: ',loss)
    return loss


def asym_unified_focal_loss(y_true,y_pred, weight=0.5, delta=0.6, gamma=0.5):
  asymmetric_ftl = asymmetric_focal_tversky_loss(y_true,y_pred, delta=delta, gamma=gamma)
  asymmetric_fl = asymmetric_focal_loss(y_true,y_pred, delta=delta, gamma=gamma)
  print('combined loss: ', asymmetric_ftl + asymmetric_fl)
  if weight is not None:
    return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)  
  else:
    return asymmetric_ftl + asymmetric_fl

              
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
    
class UnifiedFocalLoss(_WeightedLoss):
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
        super(UnifiedFocalLoss, self).__init__(weight)

    def forward(self, input, target):
        _assert_no_grad(target)
        return asym_unified_focal_loss(target,input)
           
        
def hd(inp, target, eps = 10e-5):

    inpn = inp#.numpy()
    tarn = target#.numpy()
#    print(inpn.size())
#    print(tarn.size())
    #compute hausdorff distance (HD) for all pixels outside of the tumor
    d_xy_p0 = ndi.distance_transform_edt(tarn==0)
    d_xy_p0[d_xy_p0>20] = np.max(d_xy_p0[d_xy_p0<20])
    d_xy_p0[d_xy_p0 != 0]=np.log(d_xy_p0[d_xy_p0 != 0])
#    print(d_xy_p0.sum())
#    print(d_xy_p0.max())
    #compute hausdorff distance (HD) for all pixels inside of the tumor
    d_xy_n0 = ndi.distance_transform_edt(tarn!=0)
    d_xy_n0[d_xy_n0 != 0]=np.log(d_xy_n0[d_xy_n0 != 0])
    
#    print(d_xy_n0.sum())
    #normalizing the HD's
    d_max_p =torch.where(inpn!=0,torch.from_numpy(d_xy_p0).double(),torch.zeros(inpn.size()).double()).max()+eps #normalizing by maximum HD where px is not zero
    d_xy_p = d_xy_p0/d_max_p
#    print(d_xy_p.sum())
    d_max_n = torch.where((1-inpn)!=0,torch.from_numpy(d_xy_n0).double(),torch.zeros(inpn.size()).double()).max()+eps#normalizing by maximum HD where 1-px is not zero
    d_xy_n = d_xy_n0/d_max_n
#    print(d_max_n)
#    print(d_xy_n.sum())
    #loss terms
    first_term = inpn*d_xy_p.float() #outside the tumor
    second_term = (1-inpn)*d_xy_n.float() #inside the tumor
    
    #normalizing the loss according to the paper's first term normalizing factor

#    print(torch.sum(first_term))
#    print((np.max([torch.sum(inpn[tarn==0]),eps])))
#    print(torch.sum(second_term))    
#    print(np.max([torch.sum(1-inpn[tarn!=0]),eps]))
    loss1 = torch.sum(first_term)/((np.max([torch.sum(inpn[tarn==0]),eps])))
    loss2 = torch.sum(second_term)/((np.max([torch.sum(1-inpn[tarn!=0]),eps])))
#    print(loss1)
#    print(loss2)
    return loss1+loss2
      
class HDLoss(Module):
    r"""Creates a criterion that measures the Hausdorff Distance Loss
    between the target and the output:

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If `reduce` is False, then `(N, *)`, same shape as
          input.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = HDLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """
    def __init__(self):
        super(HDLoss, self).__init__()

    def forward(self, input, target):
        _assert_no_grad(target)
        return hd(input, target)
    
def bhatt_coeff(output,target):
    out = np.sum(np.sqrt(np.abs(np.multiply(norm2validProb(output), norm2validProb(target)))))
    return out       

def bt(output,target, eps = 10e-20):
    target1 = target - torch.min(target)
    target2 = target1/torch.sum(target1)
    output1 = output - torch.min(output)
    output2 = output1/torch.sum(output1)
    out = torch.sum(torch.sqrt(torch.abs(torch.mul(output2, target2))+eps))
    return out
      
class BTLoss(Module):
    r"""Creates a criterion that measures the Batthacaryya Distance Loss
    between the target and the output:

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If `reduce` is False, then `(N, *)`, same shape as
          input.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = BTLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """
    def __init__(self):
        super(BTLoss, self).__init__()

    def forward(self, input, target):
        _assert_no_grad(target)
        return -2*bt(input, target)
        
        
def distance_metric(inp, target, eps = 10e-5):

    inpn = inp
    tarn = target
    
    tarn = tarn.byte()
    #Euclidean distance for all pixels outside and inside of the tumor
    outside = ndi.distance_transform_edt(tarn)
    inside = ndi.distance_transform_edt(~tarn)
    #signed distance
    signed_distance = torch.zeros_like(tarn)
    signed_distance = inside*~tarn - (outside-1)*tarn

    multipled = torch.mul(inpn, signed_distance.float())

    loss = multipled.mean()

    return loss


      
class DistLoss(Module):
    r"""Creates a criterion that measures Distance Loss
    between the target and the output:

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If `reduce` is False, then `(N, *)`, same shape as
          input.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = DistLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """
    def __init__(self):
        super(DistLoss, self).__init__()

    def forward(self, input, target):
        _assert_no_grad(target)
        return distance_metric(input, target)
            
def hausdist(vol_a, vol_b, thr=1, vox_spacing=(0.79323106,0.7936626,0.78976499)):
    """
    Hausdorff distance metric computation between two N-numpy array. (2D or 3D)
    The distance used is the euclidean distance.
    Args:
    * Input:
        - vol_a: N numpy array
        - vol_b: N numpy array (must be same size)
        - thr: threshold of outliers. To account for noise so it is not taken into account. Default = 1 (no tolerance). Ex. HD95%
    * Output: Scalar. Hausdorff distance between both volumes.
    
    Author: Roque
    """
    if vol_a.shape != vol_b.shape:    
        raise ValueError("Volumes must have the same dimensions")
    if vol_a.ndim == 3:
        bin_structure = np.ones((3,3,3))
    else:
        bin_structure = np.ones((3,3))
    #Computing boundaries for structures      
    contour1 = vol_a - ndi.binary_erosion(vol_a,bin_structure)
    contour2 = vol_b - ndi.binary_erosion(vol_b,bin_structure)
    #Computing distance matrices from the boundaries to both inside and outside.
    dist_a = ndi.morphology.distance_transform_edt(1-contour1, sampling=vox_spacing)
    dist_b = ndi.morphology.distance_transform_edt(1-contour2, sampling=vox_spacing)
    #Computing the infimum: The minimum distance of one volume boundary to the other
    inf_a2b = np.zeros(dist_a.shape)
    inf_a2b[dist_a==0]=dist_b[dist_a==0]
    inf_b2a = np.zeros(dist_b.shape)
    inf_b2a[dist_b==0]=dist_a[dist_b==0]
    #Compting supremum and maximum: maximum value among both cases A2B and B2A and also the supremum in it.  
    one_axis=np.r_[inf_a2b[inf_a2b>0],inf_b2a[inf_b2a>0]]
    if not one_axis.size:
        return 0
    else:
        return np.percentile(one_axis,100*thr)    
    

def hausdist_cc(vol_a, vol_b, thr=1):
    """
    Hausdorff distance metric computation between two N-numpy array. (2D or 3D)
    The distance used is the euclidean distance.
    ONLY for comparing regions with some overlap!
    Args:
    * Input:
        - vol_a: N numpy array
        - vol_b: N numpy array (must be same size)
        - thr: threshold of outliers. To account for noise so it is not taken into account. Default = 1 (no tolerance). Ex. HD95%
    * Output: Scalar. Hausdorff distance between both volumes.
    
    Author: Roque
    """
    if vol_a.shape != vol_b.shape:    
        raise ValueError("Volumes must have the same dimensions")
    if vol_a.ndim == 3:
        bin_structure = np.ones((3,3,3))
    else:
        bin_structure = np.ones((3,3))
    labelled=label(vol_a)
    vol_a=np.multiply(vol_a==1, labelled[0]==np.max(np.multiply(labelled[0],vol_b))).astype(float)
    #Computing boundaries for structures      
    contour1 = vol_a - ndi.binary_erosion(vol_a,bin_structure)
    contour2 = vol_b - ndi.binary_erosion(vol_b,bin_structure)
    #Computing distance matrices from the boundaries to both inside and outside.
    dist_a = ndi.morphology.distance_transform_edt(1-contour1)
    dist_b = ndi.morphology.distance_transform_edt(1-contour2)
    #Computing the infimum: The minimum distance of one volume boundary to the other
    inf_a2b = np.zeros(dist_a.shape)
    inf_a2b[dist_a==0]=dist_b[dist_a==0]
    inf_b2a = np.zeros(dist_b.shape)
    inf_b2a[dist_b==0]=dist_a[dist_b==0]
    #Compting supremum and maximum: maximum value among both cases A2B and B2A and also the supremum in it.  
    one_axis=np.r_[inf_a2b[inf_a2b>0],inf_b2a[inf_b2a>0]]
    if not one_axis.size:
        return 0
    else:
        return np.percentile(one_axis,100*thr)    