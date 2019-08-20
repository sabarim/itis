import random

import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt
from scipy.stats import norm
from skimage import morphology

from datasets import DataKeys
from datasets.Loader import register_dataset
from datasets.PascalVOC.PascalVOC_clicks import PascalClicksDataset
from datasets.util.DistanceTransform import get_distance_transform

VOID_LABEL = 255
NUM_CLASSES = 2
NAME = "pascalvoc_dios"

# Number of positive clicks to sample
Npos = 5
# Number of negative clicks to sample using strategy 1, 2 and 3 respectively of https://arxiv.org/abs/1603.04042
Nneg1 = 10
Nneg2 = 5
Nneg3 = 10
D = 40


@register_dataset(NAME)
class PascalVOCDiosDataset(PascalClicksDataset):
  def __init__(self, config, subset, name=NAME):
    super().__init__(config, subset, name=name)
    self.d_margin = 5
    self.strategies = [1, 2, 3]
    self.n_pairs = config.int("n_pairs", 3)
    self.neg1, self.neg2, self.neg3 = config.int_list("clicks_per_sampling_strategy", [Nneg1, Nneg2, Nneg3])

  def postproc_example_before_assembly(self, tensors):
    tensors = super().postproc_example_before_assembly(tensors)
    neg_dist_transform, pos_dist_transform, neg_clicks, pos_clicks, num_clicks = \
      tf.py_func(self.dios_distance_transform,
                 [tensors[DataKeys.SEGMENTATION_LABELS],
                  tensors[DataKeys.RAW_SEGMENTATION_LABELS],
                  [255],
                  tensors[DataKeys.IMAGE_FILENAMES]],
                 [tf.float32, tf.float32, tf.int64, tf.int64, tf.int64], name="dios_distance_transform")
    neg_dist_transform = self.set_dist_transform_shape(neg_dist_transform, tensors[DataKeys.SEGMENTATION_LABELS])
    pos_dist_transform = self.set_dist_transform_shape(pos_dist_transform, tensors[DataKeys.SEGMENTATION_LABELS])
    tensors[DataKeys.NEG_CLICKS] = neg_dist_transform
    tensors[DataKeys.POS_CLICKS] = pos_dist_transform
    return tensors

  def set_dist_transform_shape(self, dist_transform, label):
    dist_transform.set_shape(label.get_shape())
    return dist_transform

  def dios_distance_transform(self, label, raw_label, ignore_classes, img_filenames):
    strategy = random.sample([1, 2, 3], 1)[0]
    u0, neg_clicks = self.get_neg_dst_transform(label[:, :, 0], strategy, raw_label[:, :, 0], ignore_classes)
    u1, pos_clicks = self.get_pos_dst_transform(label[:, :, 0])
    num_clicks = len(neg_clicks) + len(pos_clicks)
    return u0.astype(np.float32), u1.astype(np.float32), np.array(neg_clicks).astype(np.int64), \
           np.array(pos_clicks).astype(np.int64), num_clicks

  def get_pos_dst_transform(self, label):
    label_binary = label.copy()
    label_binary[label != 1] = 0
    # Leave a margin around the object boundary
    img_area = morphology.binary_erosion(label_binary, morphology.diamond(self.d_margin))
    img_area = img_area if len(np.where(img_area == 1)[0]) > 0 else np.copy(label)

    # Set of ground truth pixels.
    O = np.where(img_area == 1)
    # Randomly sample the number of positive clicks and negative clicks to use.
    num_clicks_pos = 0 if len(O) == 0 else random.sample(range(1, Npos + 1), 1)
    # num_clicks_pos = random.sample(range(1, Npos + 1), 1)
    pts = self.get_sampled_locations(O, img_area, num_clicks_pos)
    u1 = get_distance_transform(pts, img_area)

    u1 = self.normalise(u1)

    return u1[:, :, np.newaxis], pts

  def get_neg_dst_transform(self, label, strategy, raw_label, ignore_classes):
    """
    :param raw_label: 
    :param label: 
    :param ignore_classes: 
    :param strategy: value in [1,2,3]
            1 - Generate random clicks from the background, which is D pixels away from the object.
            2 - Generate random clicks on each negative object.
            3 - Generate random clicks around the object boundary.
    :return: Negative distance transform map
    """
    g_c = self.get_image_area_to_sample(label)

    pts = []

    if strategy in [1, 3]:
      if strategy == 1:
        num_neg_clicks = random.sample(range(0, self.neg1 + 1), 1)
        pts = self.get_sampled_locations(np.where(g_c == 1), g_c, num_neg_clicks)
      else:
        # First negative click is randomly sampled in g_c
        pts = self.get_sampled_locations(np.where(g_c == 1), g_c, [1])
        g_c_copy = np.copy(g_c)
        g_c_copy[list(zip(*(val for val in pts)))] = 0
        dt = distance_transform_edt(g_c_copy)
        # Sample successive points using p_next = arg max f(p_ij | s0 U g), where p_ij in g_c, s0 is the set of all
        # sampled points, and 'g' is the complementary set of g_c
        for n_clicks in range(2, self.neg3 + 1):
          if np.max(dt) > 0:
            row, col = np.where(dt == np.max(dt))
            row, col = list(zip(row, col))[0]
            pts.append((row, col))
            x_min = max(0, row - D)
            x_max = min(row + D, dt.shape[0])
            y_min = max(0, col - D)
            y_max = min(col + D, dt.shape[1])
            dt[x_min:x_max, y_min:y_max] = 0

    elif strategy == 2:
      label_unmodified = raw_label.copy()
      label_unmodified[label != 0] = 0
      ignore_classes = np.append(ignore_classes, 0)
      # Get all negative object instances.
      instances = np.setdiff1d(np.unique(label_unmodified), ignore_classes)

      num_neg_clicks = random.sample(range(0, self.neg2 + 1), 1)
      for i in instances:
        g_c = np.zeros_like(label)
        g_c[label_unmodified == i] = 1
        pts_local = self.get_sampled_locations(np.where(g_c == 1), g_c, num_neg_clicks)
        pts = pts + pts_local

    u0 = get_distance_transform(pts, label)
    u0 = self.normalise(u0)
    return u0[:, :, np.newaxis], pts

  def normalise(self, dt):
    dt = dt.astype(np.float32)
    if self.use_gaussian:
      dt[dt > 20] = 20
      dt = norm.pdf(dt, loc=0, scale=10) * 25
    else:
      dt[dt > 255] = 255
      dt /= 255.0
    return dt

  def get_sampled_locations(self, sample_locations, img_area, num_clicks, D=40):
    d_step = int(D / 2)
    img = np.copy(img_area)
    pts = []
    for click in range(num_clicks[0]):
      pixel_samples = list(zip(sample_locations[0], sample_locations[1]))
      if len(pixel_samples) > 1:
        [x, y] = random.sample(pixel_samples, 1)[0]
        pts.append([x, y])

        x_min = max(0, x - d_step)
        x_max = min(x + d_step, img.shape[0])
        y_min = max(0, y - d_step)
        y_max = min(y + d_step, img.shape[1])
        img[x_min:x_max, y_min:y_max] = 0

        sample_locations = np.where(img == 1)

    return pts

  def get_image_area_to_sample(self, img):
    """
    calculate set g_c, which has two properties
    1) They represent background pixels 
    2) They are within a certain distance to the object
    :param img: Image that represents the object instance
    """

    # TODO: In the paper 'Deep Interactive Object Selection', they calculate g_c first based on the original object instead
    # of the dilated one.

    # Dilate the object by d_margin pixels to extend the object boundary
    img_area = np.copy(img)
    # Ignore void label for dilation
    img_area[img != 1] = 0
    img_area = morphology.binary_dilation(img_area, morphology.diamond(self.d_margin)).astype(np.uint8)

    g_c = np.logical_not(img_area).astype(int)
    g_c[distance_transform_edt(g_c) > D] = 0
    g_c[img == 255] = 0

    return g_c
