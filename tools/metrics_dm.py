from scipy import ndimage
import numpy as np
import tools.lookup_tables as lookup_tables
import scipy.ndimage as ndi


# metrics definitions
def dice_dm(mask_gt, mask_pred):
  volume_sum = mask_gt.sum() + mask_pred.sum()
  volume_intersect = (mask_gt * mask_pred).sum()
  return 2*volume_intersect / volume_sum

def hd_dm(surface_distances, percent):

  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt = surface_distances["surfel_areas_gt"]
  surfel_areas_pred = surface_distances["surfel_areas_pred"]
  if len(distances_gt_to_pred) > 0:  # pylint: disable=g-explicit-length-test
    surfel_areas_cum_gt = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
    idx = np.searchsorted(surfel_areas_cum_gt, percent/100.0)
    perc_distance_gt_to_pred = distances_gt_to_pred[
        min(idx, len(distances_gt_to_pred)-1)]
  else:
    perc_distance_gt_to_pred = np.Inf

  if len(distances_pred_to_gt) > 0:  # pylint: disable=g-explicit-length-test
    surfel_areas_cum_pred = (np.cumsum(surfel_areas_pred) /
                             np.sum(surfel_areas_pred))
    idx = np.searchsorted(surfel_areas_cum_pred, percent/100.0)
    perc_distance_pred_to_gt = distances_pred_to_gt[
        min(idx, len(distances_pred_to_gt)-1)]
  else:
    perc_distance_pred_to_gt = np.Inf

  return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)    

def msd(surface_distances):
  """Returns the average surface distance.
  Computes the average surface distances by correctly taking the area of each
  surface element into account. Call compute_surface_distances(...) before, to
  obtain the `surface_distances` dict.
  Args:
    surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
    "surfel_areas_gt", "surfel_areas_pred" created by
    compute_surface_distances()
  Returns:
    A tuple with two float values: the average distance (in mm) from the
    ground truth surface to the predicted surface and the average distance from
    the predicted surface to the ground truth surface.
  """
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt = surface_distances["surfel_areas_gt"]
  surfel_areas_pred = surface_distances["surfel_areas_pred"]
  average_distance_gt_to_pred = (
      np.sum(distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt))
  average_distance_pred_to_gt = (
      np.sum(distances_pred_to_gt * surfel_areas_pred) /
      np.sum(surfel_areas_pred))
  return (average_distance_gt_to_pred, average_distance_pred_to_gt)

def compute_surface_distances_dm(mask_gt, mask_pred, spacing_mm):
  # compute the area for all 256 possible surface elements
  # (given a 2x2x2 neighbourhood) according to the spacing_mm
  neighbour_code_to_surface_area = np.zeros([256])
  for code in range(256):
    normals = np.array(lookup_tables.neighbour_code_to_normals[code])
    sum_area = 0
    for normal_idx in range(normals.shape[0]):
      # normal vector
      n = np.zeros([3])
      n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
      n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
      n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
      area = np.linalg.norm(n)
      sum_area += area
    neighbour_code_to_surface_area[code] = sum_area

  # compute the bounding box of the masks to trim
  # the volume to the smallest possible processing subvolume
  mask_all =  np.logical_or(mask_gt, mask_pred)
  bbox_min = np.zeros(3, np.int64)
  bbox_max = np.zeros(3, np.int64)

  # max projection to the x0-axis
  proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
  idx_nonzero_0 = np.nonzero(proj_0)[0]
  if len(idx_nonzero_0) == 0:  # pylint: disable=g-explicit-length-test
    return {"distances_gt_to_pred": np.array([]),
            "distances_pred_to_gt": np.array([]),
            "surfel_areas_gt": np.array([]),
            "surfel_areas_pred": np.array([])}

  bbox_min[0] = np.min(idx_nonzero_0)
  bbox_max[0] = np.max(idx_nonzero_0)

  # max projection to the x1-axis
  proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
  idx_nonzero_1 = np.nonzero(proj_1)[0]
  bbox_min[1] = np.min(idx_nonzero_1)
  bbox_max[1] = np.max(idx_nonzero_1)

  # max projection to the x2-axis
  proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
  idx_nonzero_2 = np.nonzero(proj_2)[0]
  bbox_min[2] = np.min(idx_nonzero_2)
  bbox_max[2] = np.max(idx_nonzero_2)

  # crop the processing subvolume.
  # we need to zeropad the cropped region with 1 voxel at the lower,
  # the right and the back side. This is required to obtain the "full"
  # convolution result with the 2x2x2 kernel
  cropmask_gt = np.zeros((bbox_max - bbox_min)+2, np.uint8)
  cropmask_pred = np.zeros((bbox_max - bbox_min)+2, np.uint8)

  cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0]+1,
                                          bbox_min[1]:bbox_max[1]+1,
                                          bbox_min[2]:bbox_max[2]+1]

  cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0]+1,
                                              bbox_min[1]:bbox_max[1]+1,
                                              bbox_min[2]:bbox_max[2]+1]

  # compute the neighbour code (local binary pattern) for each voxel
  # the resultsing arrays are spacially shifted by minus half a voxel in each
  # axis.
  # i.e. the points are located at the corners of the original voxels
  kernel = np.array([[[128, 64],
                      [32, 16]],
                     [[8, 4],
                      [2, 1]]])
  neighbour_code_map_gt = ndimage.filters.correlate(
      cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
  neighbour_code_map_pred = ndimage.filters.correlate(
      cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

  # create masks with the surface voxels
  borders_gt = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
  borders_pred = ((neighbour_code_map_pred != 0) &
                  (neighbour_code_map_pred != 255))

  # compute the distance transform (closest distance of each voxel to the
  # surface voxels)
  if borders_gt.any():
    distmap_gt = ndimage.morphology.distance_transform_edt(
        ~borders_gt, sampling=spacing_mm)
  else:
    distmap_gt = np.Inf * np.ones(borders_gt.shape)

  if borders_pred.any():
    distmap_pred = ndimage.morphology.distance_transform_edt(
        ~borders_pred, sampling=spacing_mm)
  else:
    distmap_pred = np.Inf * np.ones(borders_pred.shape)

  # compute the area of each surface element
  surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
  surface_area_map_pred = neighbour_code_to_surface_area[
      neighbour_code_map_pred]

  # create a list of all surface elements with distance and area
  distances_gt_to_pred = distmap_pred[borders_gt]
  distances_pred_to_gt = distmap_gt[borders_pred]
  surfel_areas_gt = surface_area_map_gt[borders_gt]
  surfel_areas_pred = surface_area_map_pred[borders_pred]

  # sort them by distance
  if distances_gt_to_pred.shape != (0,):
    sorted_surfels_gt = np.array(
        sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
    distances_gt_to_pred = sorted_surfels_gt[:, 0]
    surfel_areas_gt = sorted_surfels_gt[:, 1]

  if distances_pred_to_gt.shape != (0,):
    sorted_surfels_pred = np.array(
        sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
    distances_pred_to_gt = sorted_surfels_pred[:, 0]
    surfel_areas_pred = sorted_surfels_pred[:, 1]

  return {"distances_gt_to_pred": distances_gt_to_pred,
          "distances_pred_to_gt": distances_pred_to_gt,
          "surfel_areas_gt": surfel_areas_gt,
          "surfel_areas_pred": surfel_areas_pred}
    