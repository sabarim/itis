import glob
import os
import pickle

import numpy as np
import tensorflow as tf

from core import Extractions
from core.Log import log
from datasets import DataKeys
from datasets.Loader import register_dataset
from datasets.PascalVOC.PascalVOC_dios import PascalVOCDiosDataset
from datasets.util.DistanceTransform import get_distance_transform, unsigned_distance_transform
from datasets.util.Util import encodeMask, decodeMask, generate_click_for_correction

NAME = "pascalvoc_masked_dios"
PREV_MASK = "previous_mask"
PREV_NEG_CLICKS = "previous_neg_clicks"
PREV_POS_CLICKS = "previous_pos_clicks"
POSTERIORS = "posteriors"
# Probability that the mask from previous iteration is removed.
P = 0.3


@register_dataset(NAME)
class PascalVOCMaskedDiosDataset(PascalVOCDiosDataset):
  def __init__(self, config, subset):
    super().__init__(config, subset, name=NAME)
    self.previous_epoch_data = {}
    self.iterative_training = config.bool("iterative_training", True)
    self.use_posteriors = config.bool("use_posteriors", False)
    self.initial_mask_probability = config.float("initial_mask_probability", 0.5)
    self.click_reset_probability = config.float("click_reset_probability", P)
    self.initialise_with_single_click = config.bool("initialise_with_single_click", False)
    self.model = config.string("model")
    self.model_base_dir = config.dir("model_dir", "models")
    self.model_dir = self.model_base_dir + self.model + "/"
    self.load_old_label_dict()
    self.ignore_previous_mask = np.repeat([False], self.n_examples_per_epoch())

    if self.subset == "train":
      self.p = self.click_reset_probability
    else:
      self.p = 0.0

  def load_old_label_dict(self):
    files = sorted(glob.glob(self.model_dir + self.model + "-*.pickle"))
    if len(files) > 0:
      self.previous_epoch_data = pickle.load(open(files[-1], 'rb'))

  def   postproc_example_before_assembly(self, tensors):
    # Flag to ignore previous mask based on a probability which is a hyper parameter.
    self.ignore_previous_mask = np.random.choice([True, False], self.n_examples_per_epoch(), p=[self.p, 1 - self.p])
    tensors = super().postproc_example_before_assembly(tensors)
    assert DataKeys.IMAGE_FILENAMES in tensors
    tensors[DataKeys.BBOX_GUIDANCE] = tf.py_func(self.get_previous_mask, [tensors[DataKeys.IMAGE_FILENAMES],
                                                                          tensors[DataKeys.SEGMENTATION_LABELS]],
                                                 tf.float32, name="get_previous_mask")
    tensors[DataKeys.BBOX_GUIDANCE].set_shape(tensors[DataKeys.SEGMENTATION_LABELS].get_shape())

    if self.use_unsigned_distance_transform_guidance:
      bbox_guidance = tensors[DataKeys.BBOX_GUIDANCE]
      udt = unsigned_distance_transform(bbox_guidance)
      tensors[DataKeys.UNSIGNED_DISTANCE_TRANSFORM_GUIDANCE] = udt

    return tensors

  def get_previous_mask(self, img_filename, label):
    if img_filename in self.previous_epoch_data and PREV_MASK in self.previous_epoch_data[img_filename] and \
       not self.is_ignore(img_filename):
      mask = decodeMask(self.previous_epoch_data[img_filename][PREV_MASK])[:, :, np.newaxis] if not self.use_posteriors \
              else np.expand_dims(self.previous_epoch_data[img_filename][POSTERIORS], axis=2)
    elif self.use_posteriors:
      mask = np.ones_like(label) * self.initial_mask_probability
    elif self.use_unsigned_distance_transform_guidance:
      mask = np.ones_like(label)
    else:
      mask = np.zeros_like(label)

    return mask.astype(np.float32)

  def use_segmentation_mask(self, res):
    extractions = res[Extractions.EXTRACTIONS]

    # if self.subset == "train" and self.iterative_training:
    if self.iterative_training:
      assert DataKeys.IMAGE_FILENAMES in extractions
      assert Extractions.SEGMENTATION_MASK_INPUT_SIZE in extractions
      batch_size = len(extractions[DataKeys.IMAGE_FILENAMES][0])

      for id in range(batch_size):
        filename = extractions[DataKeys.IMAGE_FILENAMES][0][id]
        self.previous_epoch_data[filename][PREV_MASK] = \
          encodeMask(extractions[Extractions.SEGMENTATION_MASK_INPUT_SIZE][0][id])
        if self.use_posteriors:
          self.previous_epoch_data[filename][POSTERIORS] = extractions[Extractions.SEGMENTATION_POSTERIORS][0][id]

  def dios_distance_transform(self, label, raw_label, ignore_classes, img_filenames):
    if img_filenames not in self.previous_epoch_data or self.is_ignore(img_filenames):
      # initialisation with single click as a 4th strategy.
      neg4 = np.random.choice([True, False], p=[self.p, 1 - self.p])
      if self.initialise_with_single_click or neg4:
        print("Using single click initialisation", file=log.v1)
        self.previous_epoch_data[img_filenames] = {}
        self.previous_epoch_data[img_filenames][PREV_NEG_CLICKS] = []
        self.previous_epoch_data[img_filenames][PREV_POS_CLICKS] = []
        neg_clicks, num_clicks, pos_clicks, u0, u1 = self.distance_transform_from_prev_mask(img_filenames, label)
      else:
        u0, u1, neg_clicks, pos_clicks, num_clicks = super(). \
          dios_distance_transform(label, raw_label, ignore_classes, img_filenames)
        self.previous_epoch_data[img_filenames] = {}
        self.previous_epoch_data[img_filenames][PREV_NEG_CLICKS] = neg_clicks.tolist()
        self.previous_epoch_data[img_filenames][PREV_POS_CLICKS] = pos_clicks.tolist()
    else:
      neg_clicks, num_clicks, pos_clicks, u0, u1 = self.distance_transform_from_prev_mask(img_filenames, label)

    # Sanity checks
    if np.any(label[np.where(u0[:, :, 0] == 0)]):
      print("Neg clicks on the object detected", label[np.where(u0[:, :, 0] == 0)], file=log.v1)
    if not np.all(label[np.where(u1[:, :, 0] == 0)]):
      print("Pos clicks on background detected", file=log.v1)

    return u0.astype(np.float32), u1.astype(np.float32),\
           np.array(neg_clicks).astype(np.int64), np.array(pos_clicks).astype(np.int64), num_clicks

  def distance_transform_from_prev_mask(self, img_filenames, label):
    neg_clicks, pos_clicks, neg_click_new, pos_click_new = self.sample_new_clicks(img_filenames, label)
    neg_clicks += neg_click_new
    pos_clicks += pos_click_new
    u0 = self.normalise(get_distance_transform(neg_clicks, label))
    u1 = self.normalise(get_distance_transform(pos_clicks, label))
    self.previous_epoch_data[img_filenames][PREV_NEG_CLICKS] = neg_clicks
    self.previous_epoch_data[img_filenames][PREV_POS_CLICKS] = pos_clicks
    num_clicks = len(pos_clicks + neg_clicks)
    return neg_clicks, num_clicks, pos_clicks, u0, u1

  def sample_new_clicks(self, img_filenames, label):
    neg_clicks = self.previous_epoch_data[img_filenames][PREV_NEG_CLICKS]
    pos_clicks = self.previous_epoch_data[img_filenames][PREV_POS_CLICKS]
    neg_clicks_new = []
    pos_clicks_new = []

    if len(label.shape) > 2:
      label = label.copy()
      label = label[:, :, 0]

    if PREV_MASK not in self.previous_epoch_data[img_filenames]:
      prediction = np.zeros_like(label)
    else:
      prediction = decodeMask(self.previous_epoch_data[img_filenames][PREV_MASK])

    clicks = generate_click_for_correction(label, prediction, neg_clicks + pos_clicks, void_label=255)
    if len(clicks) > 0:
      if label[clicks[0][0], clicks[0][1]] == 1:
        pos_clicks_new += clicks
      else:
        neg_clicks_new += clicks

    return neg_clicks, pos_clicks, neg_clicks_new, pos_clicks_new

  def get_extraction_keys(self):
    return [Extractions.SEGMENTATION_MASK_INPUT_SIZE, DataKeys.IMAGE_FILENAMES, Extractions.SEGMENTATION_POSTERIORS]

  def is_ignore(self, filename):
    if filename in self.previous_epoch_data and self.subset == "train":
      keys = list(self.previous_epoch_data.keys())
      return self.ignore_previous_mask[keys.index(filename)]
    else:
      return True

  def save_masks(self, epoch):
    if not os.path.exists(self.model_dir):
      os.mkdir(self.model_dir)

    fn = self.model_dir + self.model + "-" + str(epoch) + ".pickle"
    pickle.dump(self.previous_epoch_data, open(fn, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
