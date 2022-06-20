"""
This script aims to take the box predictions, put them in the original size and put them back into the 
"""

import os
import numpy as np
from misc import get_boxes_original_size

path_to_experiments = 'D:/project_2/scripts/p2_segmentation/Experiments_as/'

# adding this segm_results_npy_test

list_of_experiments = ['box_gen_loss_fold_1','box_gen_loss_fold_2','box_gen_loss_fold_3','box_gen_loss_fold_4']

# paths_to_outs = [os.path.join(path_to_experiments,x,'segm_results_npy_test') for x in list_of_experiments]
# paths_to_big_boxes = [os.path.join(path_to_experiments,x,'boxes') for x in list_of_experiments]
# paths_to_thumb = [os.path.join(path_to_experiments,x,'thumb_boxes') for x in list_of_experiments]

for exp in list_of_experiments:
    this_path_out=os.path.join(path_to_experiments,exp,'segm_results_npy_test/') 
    this_path_boxes=os.path.join(path_to_experiments,exp,'boxes/') 
    this_path_thumb=os.path.join(path_to_experiments,exp,'thumb_boxes/') 
    # consider automatic saving of the dir tree.
    get_boxes_original_size(this_path_out,this_path_boxes,this_path_thumb)