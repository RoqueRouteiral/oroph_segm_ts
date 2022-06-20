import numpy as np
import scipy.ndimage as ndi
from skimage.transform import PiecewiseAffineTransform, warp


## Oficial keras functions
#functions from https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py

#def gray_level_transform(img, mean, std):
#    # create transform.
#    transf = 
#    # Apply transform. 
#    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8') 
#    return img

def random_rotation(x, y, rg, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.deg2rad(np.random.uniform(-rg, rg))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_axis, fill_mode, cval)
    return x, y
    
def random_rotation2(x, y, z, rg, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.deg2rad(np.random.uniform(-rg, rg))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_axis, fill_mode, cval)
    z = apply_transform(z, transform_matrix, channel_axis, fill_mode, cval)
    return x, y, z    

def random_rotation3(x, y, z, t, rg, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.deg2rad(np.random.uniform(-rg, rg))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_axis, fill_mode, cval)
    z = apply_transform(z, transform_matrix, channel_axis, fill_mode, cval)
    t = apply_transform(t, transform_matrix, channel_axis, fill_mode, cval)
    return x, y, z, t 

def random_rotation4(x, y, z, t, s, rg, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.deg2rad(np.random.uniform(-rg, rg))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_axis, fill_mode, cval)
    z = apply_transform(z, transform_matrix, channel_axis, fill_mode, cval)
    t = apply_transform(t, transform_matrix, channel_axis, fill_mode, cval)
    s = apply_transform(s, transform_matrix, channel_axis, fill_mode, cval)
    return x, y, z, t, s 

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Applies the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def horizontal_flip(x,y):
    X = flip_axis(x, 1)
    Y = flip_axis(y, 1)
    return X, Y
#
def vertical_flip(x,y):
    X = flip_axis(x, 0)
    Y = flip_axis(y, 0)
    return X, Y

def horizontal_flip3(x,y,z,t):
    X = flip_axis(x, 1)
    Y = flip_axis(y, 1)
    Z = flip_axis(z, 1)
    T = flip_axis(t, 1)    
    return X, Y, Z, T
#
def vertical_flip3(x,y,z,t):
    X = flip_axis(x, 0)
    Y = flip_axis(y, 0)
    Z = flip_axis(z, 0)
    T = flip_axis(t, 0)    
    return X, Y, Z, T

def horizontal_flip4(x,y,z,t,w):
    X = flip_axis(x, 1)
    Y = flip_axis(y, 1)
    Z = flip_axis(z, 1)
    T = flip_axis(t, 1)
    W = flip_axis(w, 1)    
    return X, Y, Z, T, W
#
def vertical_flip4(x,y,z,t,w):
    X = flip_axis(x, 0)
    Y = flip_axis(y, 0)
    Z = flip_axis(z, 0)
    T = flip_axis(t, 0)
    W = flip_axis(w, 0)    
    return X, Y, Z, T, W
   
def flip_axis(x, axis):#check
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))
    

def elastic_deformation(image, gt, def_factor=5, grid_factor = 10):
    out = np.zeros(image.shape)
    out_gt = np.zeros(image.shape)
    rows, cols = image.shape[0], image.shape[1]
    
    src_cols = np.linspace(0, cols, grid_factor)
    src_rows = np.linspace(0, rows, grid_factor)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    
    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - (0.5-np.random.rand(grid_factor*grid_factor))*def_factor
    dst_cols = src[:, 0] - (0.5-np.random.rand(grid_factor*grid_factor))*def_factor
    
    dst = np.vstack([dst_cols, dst_rows]).T
    
    
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    
    out_rows = image.shape[0]
    out_cols = cols

    for i in range(image.shape[2]):
        out[:,:,i] = warp(image[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)#order is important
        out_gt[:,:,i] = warp(gt[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)
    return out, out_gt
       
def elastic_deformation2(image, image2, gt, def_factor=5, grid_factor = 10):
    out = np.zeros(image.shape)
    out2 = np.zeros(image2.shape)
    out_gt = np.zeros(image.shape)
    rows, cols = image.shape[0], image.shape[1]
    
    src_cols = np.linspace(0, cols, grid_factor)
    src_rows = np.linspace(0, rows, grid_factor)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    
    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - (0.5-np.random.rand(grid_factor*grid_factor))*def_factor
    dst_cols = src[:, 0] - (0.5-np.random.rand(grid_factor*grid_factor))*def_factor
    
    dst = np.vstack([dst_cols, dst_rows]).T
    
    
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    
    out_rows = image.shape[0]
    out_cols = cols

    for i in range(image.shape[2]):
        out[:,:,i] = warp(image[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)#order is important
        out2[:,:,i] = warp(image2[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)
        out_gt[:,:,i] = warp(gt[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)
    return out, out2, out_gt

def elastic_deformation3(image, image2, image3, gt, def_factor=5, grid_factor = 10):
    out = np.zeros(image.shape)
    out2 = np.zeros(image2.shape)
    out3 = np.zeros(image3.shape)
    out_gt = np.zeros(image.shape)
    rows, cols = image.shape[0], image.shape[1]
    
    src_cols = np.linspace(0, cols, grid_factor)
    src_rows = np.linspace(0, rows, grid_factor)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    
    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - (0.5-np.random.rand(grid_factor*grid_factor))*def_factor
    dst_cols = src[:, 0] - (0.5-np.random.rand(grid_factor*grid_factor))*def_factor
    
    dst = np.vstack([dst_cols, dst_rows]).T
    
    
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    
    out_rows = image.shape[0]
    out_cols = cols

    for i in range(image.shape[2]):
        out[:,:,i] = warp(image[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)#order is important
        out2[:,:,i] = warp(image2[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)
        out3[:,:,i] = warp(image3[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)
        out_gt[:,:,i] = warp(gt[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)
    return out, out2, out3, out_gt

def elastic_deformation4(image, image2, image3, image4, gt, def_factor=5, grid_factor = 10):
    out = np.zeros(image.shape)
    out2 = np.zeros(image2.shape)
    out3 = np.zeros(image3.shape)
    out4 = np.zeros(image4.shape)
    out_gt = np.zeros(image.shape)
    rows, cols = image.shape[0], image.shape[1]
    
    src_cols = np.linspace(0, cols, grid_factor)
    src_rows = np.linspace(0, rows, grid_factor)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    
    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - (0.5-np.random.rand(grid_factor*grid_factor))*def_factor
    dst_cols = src[:, 0] - (0.5-np.random.rand(grid_factor*grid_factor))*def_factor
    
    dst = np.vstack([dst_cols, dst_rows]).T
    
    
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    
    out_rows = image.shape[0]
    out_cols = cols

    for i in range(image.shape[2]):
        out[:,:,i] = warp(image[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)#order is important
        out2[:,:,i] = warp(image2[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)
        out3[:,:,i] = warp(image3[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)
        out4[:,:,i] = warp(image4[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)
        out_gt[:,:,i] = warp(gt[:,:,i], tform, output_shape=(out_rows, out_cols),order=0)
    return out, out2, out3, out4, out_gt
                           