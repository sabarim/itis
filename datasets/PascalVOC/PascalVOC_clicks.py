import tensorflow as tf
from datasets import DataKeys
from datasets.PascalVOC.PascalVOC_instance import PascalVOCInstanceDataset
from datasets.util.Normalization import unnormalize
from datasets.util.Util import visualise_clicks

NAME="pascalvoc_clicks"


class PascalClicksDataset(PascalVOCInstanceDataset):
  def __init__(self, config, subset, name=NAME):
    super().__init__(config, subset, name=name)
    self.use_gaussian = config.bool("use_gaussian", False)

  def create_summaries(self, data):
    if DataKeys.IMAGES in data:
      images = unnormalize(data[DataKeys.IMAGES])
      # TODO: Fix visualisation of clicks for gaussians.
      if DataKeys.NEG_CLICKS in data:
        # images = tf.py_func(visualise_clicks, [images, data[DataKeys.NEG_CLICKS][:, :, :, 0:1], "r"], tf.float32)
        self.summaries.append(tf.summary.image(self.subset + "data/neg_clicks",
                                               tf.cast(data[DataKeys.NEG_CLICKS][:, :, :, 0:1], tf.float32)))
      if DataKeys.POS_CLICKS in data:
        # images = tf.py_func(visualise_clicks, [images, data[DataKeys.POS_CLICKS][:, :, :, 0:1], "g"], tf.float32)
        self.summaries.append(tf.summary.image(self.subset + "data/pos_clicks",
                                               tf.cast(data[DataKeys.POS_CLICKS][:, :, :, 0:1], tf.float32)))
      self.summaries.append(tf.summary.image(self.subset + "data/images", images))

    if DataKeys.SEGMENTATION_LABELS in data:
      self.summaries.append(tf.summary.image(self.subset + "data/ground truth segmentation labels",
                                             tf.cast(data[DataKeys.SEGMENTATION_LABELS], tf.float32)))
    if DataKeys.BBOX_GUIDANCE in data:
      self.summaries.append(tf.summary.image(self.subset + "data/bbox guidance",
                                             tf.cast(data[DataKeys.BBOX_GUIDANCE], tf.float32)))
    if DataKeys.SIGNED_DISTANCE_TRANSFORM_GUIDANCE in data:
      self.summaries.append(tf.summary.image(self.subset + "data/signed_distance_transform_guidance",
                                             data[DataKeys.SIGNED_DISTANCE_TRANSFORM_GUIDANCE]))
    if DataKeys.UNSIGNED_DISTANCE_TRANSFORM_GUIDANCE in data:
      self.summaries.append(tf.summary.image(self.subset + "data/unsigned_distance_transform_guidance",
                                             data[DataKeys.UNSIGNED_DISTANCE_TRANSFORM_GUIDANCE]))