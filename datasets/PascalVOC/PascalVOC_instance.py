import os
import numpy as np
import scipy.io
from PIL import Image
import tensorflow as tf
from multiprocessing import Pool
from functools import partial

from datasets.Loader import register_dataset
from datasets import DataKeys
from datasets.PascalVOC.PascalVOC import PascalVOCDataset
from datasets.util.Util import get_filename_without_extension
from core.Timer import Timer
from core.Log import log


VOID_LABEL = 255
NUM_CLASSES = 2
NAME = "pascalvoc_instance"


@register_dataset(NAME)
class PascalVOCInstanceDataset(PascalVOCDataset):
  def __init__(self, config, subset, name=NAME):
    super().__init__(config, name, subset, NUM_CLASSES)
    self._instance_masks = {}
    self._raw_masks = {}
    self.min_area = 0

  def load_image(self, img_filename):
    # remove instance id from filename
    img_filename = tf.string_split([img_filename], ':').values[0]
    return super().load_image(img_filename)

  def load_annotation(self, img, img_filename, annotation_filename):
    mask, raw_mask = tf.py_func(self._get_mask, [img_filename], [tf.uint8, tf.uint8])
    mask.set_shape((None, None, 1))
    raw_mask.set_shape((None, None, 1))
    return {DataKeys.SEGMENTATION_LABELS: mask, DataKeys.RAW_SEGMENTATION_LABELS: raw_mask,
            DataKeys.IMAGE_FILENAMES: img_filename}

  def _get_mask(self, img_path):
    img_path = img_path.decode('utf-8')
    mask = self._instance_masks[img_path]
    raw_mask = self._raw_masks[img_path]
    return mask, raw_mask

  def read_inputfile_lists(self):
    data_list = "train.txt" if self.subset == "train" else "val.txt"
    data_list = "datasets/PascalVOC/" + data_list
    imgs = []
    ans = []
    print("PascalVOCInstanceDataset ({}): loading instances...".format(self.subset), file=log.v4)
    timer = Timer()
    with open(data_list) as f:
      lines = f.readlines()
    do_line = partial(get_instances_for_line, data_dir=self.data_dir, subset=self.subset, min_area=self.min_area)
    with Pool(4) as pool:
      img_an_ids_masks_list = pool.map(do_line, lines)
    for im, an, instance_ids, instance_masks, raw_masks in img_an_ids_masks_list:
      file_name_without_ext = get_filename_without_extension(an)
      an = self.data_dir + "/SegmentationObject/" + file_name_without_ext + ".png"
      for id_, mask, raw_mask in zip(instance_ids, instance_masks, raw_masks):
        img_path = im + ":" + str(id_)
        imgs.append(img_path)
        ans.append(an)
        self._instance_masks[img_path] = mask
        self._raw_masks[img_path] = raw_mask
    print("PascalVOCInstanceDataset ({}): Time taken to load instances: ".format(self.subset), timer.elapsed(),
          file=log.v4)
    return imgs, ans


def get_instances_for_line(line, data_dir, subset, min_area):
  im, an = line.strip().split()
  im = data_dir + im
  instance_ids, instance_masks, raw_masks = get_instances(im, data_dir, subset, min_area)
  return im, an, instance_ids, instance_masks, raw_masks


def get_instances(im, data_dir, subset, min_area):
  instance_ids = []
  instance_masks = []
  raw_masks = []
  instance_segm = read_label(data_dir, im, subset)

  if instance_segm is not None:
    inst_labels = np.unique(instance_segm)
    inst_labels = np.setdiff1d(inst_labels, [0, VOID_LABEL])

    for inst in inst_labels:
      mask = instance_segm == inst
      area = mask.sum()
      if area > min_area:
        instance_ids.append(inst)
        label = mask.astype("uint8")[:, :, np.newaxis]
        label[instance_segm == VOID_LABEL] = VOID_LABEL
        raw_label = instance_segm.astype("uint8")[:, :, np.newaxis]
        instance_masks.append(label)
        raw_masks.append(raw_label)

  return instance_ids, instance_masks, raw_masks


def read_label(data_dir, im, subset):
  instance_segm = None
  file_name_without_ext = get_filename_without_extension(im)
  inst_path = data_dir + "inst/" + file_name_without_ext + ".mat"
  # Get instances from SBD during training, if they are available
  if subset == "train" and os.path.exists(inst_path):
    instance_segm = scipy.io.loadmat(inst_path)['GTinst']['Segmentation'][0][0]
  else:
    file_name_without_ext = get_filename_without_extension(im)
    inst_path = data_dir + "/SegmentationObject/" + file_name_without_ext + ".png"
    if os.path.exists(inst_path):
      instance_segm = np.array(Image.open(inst_path))
    else:
      print("File: " + im + " does not have any instance annotations.", file=log.v1)
  return instance_segm
