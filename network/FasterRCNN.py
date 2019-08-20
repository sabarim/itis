# Cf. https://github.com/ppwwyyxx/tensorpack/blob/master/examples/FasterRCNN/train.py
# and https://github.com/ppwwyyxx/tensorpack/blob/master/examples/FasterRCNN/model.py

import tensorflow as tf
import numpy as np
import cv2

from datasets import DataKeys
from network.ConvolutionalLayers import Conv, ConvTranspose
from network.FullyConnected import FullyConnected
from network.Layer import Layer
from network.Util import prepare_input
from network.Resnet import pretrained_resnet50_conv4, add_resnet_conv5
from core import Measures
from network.FasterRCNN_utils import decode_bbox_target, encode_bbox_target,\
  generate_rpn_proposals, sample_fast_rcnn_targets, roi_align, rpn_losses,\
  fastrcnn_losses, clip_boxes, fastrcnn_predictions, maskrcnn_loss, crop_and_resize
from datasets.util.Detection import ALL_ANCHORS, NUM_ANCHOR, ANCHOR_STRIDE


FASTRCNN_BBOX_REG_WEIGHTS = np.array([10, 10, 5, 5], dtype='float32')


def rpn_head(featuremap, channel, num_anchors, tower_setup):
  with tf.variable_scope('rpn'):
    # TODO all Convs here should use kernel_initializer=tf.random_normal_initializer(stddev=0.01)
    hidden = Conv('conv0', [featuremap], channel, tower_setup, old_order=True, bias=True).outputs[0]

    label_logits = Conv('class', [hidden], num_anchors, tower_setup, (1, 1), old_order=True, bias=True, activation="linear").outputs[0]
    box_logits = Conv('box', [hidden], 4 * num_anchors, tower_setup, (1, 1), old_order=True, bias=True, activation="linear").outputs[0]
    shp = tf.shape(box_logits)
    box_logits = tf.reshape(box_logits, tf.stack([shp[0], shp[1], shp[2], num_anchors, 4]))
  return label_logits, box_logits


def fastrcnn_head(feature, num_classes, tower_setup):
  with tf.variable_scope('fastrcnn'):
    # GlobalAvgPooling, see https://tensorpack.readthedocs.io/en/latest/_modules/tensorpack/models/pool.html
    assert feature.shape.ndims == 4
    feature = tf.reduce_mean(feature, [1, 2], name='gap/output')

    # TODO this should use tf.random_normal_initializer(stddev=0.01)
    classification = FullyConnected("class", [feature], num_classes, tower_setup, activation="linear").outputs[0]
    # TODO this should use tf.random_normal_initializer(stddev=0.001)
    box_regression = FullyConnected("box", [feature], (num_classes - 1) * 4, tower_setup, activation="linear").outputs[0]
    box_regression = tf.reshape(box_regression, (-1, num_classes - 1, 4))
    return classification, box_regression


def maskrcnn_head(feature, num_class, tower_setup):
  with tf.variable_scope('maskrcnn'):
    # TODO both operations should have kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out', distribution='normal')
    # c2's MSRAFill is fan_out
    l = ConvTranspose('deconv', [feature], 256, tower_setup, (2, 2), strides=(2, 2), bias=True).outputs[0]
    l = Conv('conv', [l], num_class - 1, tower_setup, (1, 1), old_order=True, bias=True, activation="linear").outputs[0]
  return l


# See also https://github.com/ppwwyyxx/tensorpack/blob/master/examples/FasterRCNN/train.py
class FasterRCNN(Layer):
  def _get_anchors(self, shape2d):
    """
    Returns:
        FSxFSxNAx4 anchors,
    """
    # FSxFSxNAx4 (FS=MAX_SIZE//ANCHOR_STRIDE)
    with tf.name_scope('anchors'):
      all_anchors = tf.constant(ALL_ANCHORS, name='all_anchors', dtype=tf.float32)
      fm_anchors = tf.slice(
        all_anchors, [0, 0, 0, 0], tf.stack([
          shape2d[0], shape2d[1], -1, -1]), name='fm_anchors')
    return fm_anchors

  @staticmethod
  def fill_full_mask(boxes, masks, img_shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    """
    n_boxes = boxes.shape[0]
    assert n_boxes == masks.shape[0]
    ret = np.zeros([n_boxes, img_shape[0], img_shape[1]], dtype='uint8')
    for idx in range(n_boxes):
      # int() is floor
      # box fpcoor=0.0 -> intcoor=0.0
      x0, y0 = list(map(int, boxes[idx, :2] + 0.5))
      # box fpcoor=h -> intcoor=h-1, inclusive
      x1, y1 = list(map(int, boxes[idx, 2:] - 0.5))  # inclusive
      x1 = max(x0, x1)  # require at least 1x1
      y1 = max(y0, y1)

      w = x1 + 1 - x0
      h = y1 + 1 - y0

      # rounding errors could happen here, because masks were not originally computed for this shape.
      # but it's hard to do better, because the network does not know the "original" scale
      mask = (cv2.resize(masks[idx, :, :], (w, h)) > 0.5).astype('uint8')
      ret[idx, y0:y1 + 1, x0:x1 + 1] = mask

    return ret

  def __init__(self, name, inputs, network_input_dict, tower_setup, fastrcnn_batch_per_img=256):
    super(FasterRCNN, self).__init__()
    self.is_training = tower_setup.is_training
    inputs = inputs[0]
    if self.is_training:
      tf.add_to_collection('checkpoints', inputs)
    self._image_shape2d = tf.shape(network_input_dict[DataKeys.IMAGES])[1:3]
    self._add_maskrcnn = tower_setup.dataset.config.bool("add_masks", True)
    max_size = tower_setup.dataset.config.int_list("input_size_train", [])[1]
    self.bbox_decode_clip = np.log(max_size / 16.0)
    self.fastrcnn_batch_per_img = fastrcnn_batch_per_img
    self.tower_setup = tower_setup
    self.network_input_dict = network_input_dict

    with tf.variable_scope(name):
      rpn_label_logits, rpn_box_logits = rpn_head(inputs, 1024, NUM_ANCHOR, self.tower_setup)
      fm_shape = tf.shape(inputs)[1:3]  # h,w
      fm_anchors = self._get_anchors(fm_shape)

    losses = []
    batch_size = inputs.get_shape().as_list()[0]
    assert batch_size is not None
    for batch_idx in range(batch_size):
      with tf.variable_scope(name, reuse=True if batch_idx > 0 else None):
        final_boxes, final_labels, final_masks, final_probs, \
          fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss = \
          self._create_heads(batch_idx, fm_anchors, fm_shape, rpn_box_logits, inputs, rpn_label_logits)
        if self.is_training:
          losses.extend([fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss])
        else:
          self._add_test_measures(batch_idx, final_boxes, final_probs, final_labels, final_masks)
        # combine individual losses for summaries create summaries (atm they are separate)
        self.add_scalar_summary(rpn_label_loss, "rpn_label_loss")
        self.add_scalar_summary(rpn_box_loss, "rpn_box_loss")
        self.add_scalar_summary(fastrcnn_label_loss, "fastrcnn_label_loss")
        self.add_scalar_summary(fastrcnn_box_loss, "fastrcnn_box_loss")

    if self.is_training:
      vars_to_regularize = tf.trainable_variables("frcnn/(?:rpn|fastrcnn|maskrcnn)/.*W")
      regularizers = [1e-4 * tf.nn.l2_loss(W) for W in vars_to_regularize]
      regularization_loss = tf.add_n(regularizers, "regularization_loss")
      self.regularizers.append(regularization_loss)
      loss = tf.add_n(losses, 'total_cost') / batch_size
      self.losses.append(loss)
      # self.add_scalar_summary(regularization_loss, "regularization_loss")
    else:
      loss = 0.0
    self.add_scalar_summary(loss, "loss")
    self._add_basic_measures(inputs, loss)

  def _create_heads(self, batch_idx, fm_anchors, fm_shape, rpn_box_logits, rpn_input, rpn_label_logits):
    # Prepare the data, slice inputs by batch index
    gt_boxes = self.network_input_dict[DataKeys.BBOXES_x0y0x1y1]
    gt_boxes = gt_boxes[batch_idx]
    gt_labels = self.network_input_dict[DataKeys.CLASSES]
    gt_labels = tf.cast(gt_labels[batch_idx], dtype=tf.int64)
    if self._add_maskrcnn:
      gt_masks = self.network_input_dict[DataKeys.SEGMENTATION_MASK]
      gt_masks = gt_masks[batch_idx]
      gt_masks = tf.transpose(gt_masks, [2, 0, 1])
    else:
      gt_masks = None
    target_ids = self.network_input_dict[DataKeys.IDS]  # If 0, ignore
    target_ids = target_ids[batch_idx]
    gt_boxes = tf.boolean_mask(gt_boxes, tf.greater(target_ids, 0))
    gt_labels = tf.boolean_mask(gt_labels, tf.greater(target_ids, 0))
    featuremap_labels = self.network_input_dict[DataKeys.FEATUREMAP_LABELS]
    featuremap_labels = featuremap_labels[batch_idx]
    featuremap_boxes = self.network_input_dict[DataKeys.FEATUREMAP_BOXES]  # already in xyxy format
    featuremap_boxes = featuremap_boxes[batch_idx]
    rpn_label_logits = rpn_label_logits[batch_idx]
    rpn_box_logits = rpn_box_logits[batch_idx]

    decoded_boxes = decode_bbox_target(self.bbox_decode_clip, rpn_box_logits, fm_anchors)  # fHxfWxNAx4, floatbox
    proposal_boxes, proposal_scores = generate_rpn_proposals(
      tf.reshape(decoded_boxes, [-1, 4]),
      tf.reshape(rpn_label_logits, [-1]),
      self._image_shape2d, self.is_training)
    if self.is_training:
      # sample proposal boxes in training
      rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
        proposal_boxes, gt_boxes, gt_labels, self.fastrcnn_batch_per_img)
      boxes_on_featuremap = rcnn_sampled_boxes * (1.0 / ANCHOR_STRIDE)
      #tf.add_to_collection('checkpoints', boxes_on_featuremap)
    else:
      rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt = None, None, None
      # use all proposal boxes in inference
      boxes_on_featuremap = proposal_boxes * (1.0 / ANCHOR_STRIDE)
    fastrcnn_box_logits, fastrcnn_label_logits, feature_fastrcnn = self._create_fastrcnn_output(boxes_on_featuremap,
                                                                                                rpn_input)
    if self.is_training:
      #tf.add_to_collection('checkpoints', fastrcnn_box_logits)
      #tf.add_to_collection('checkpoints', fastrcnn_label_logits)
      #tf.add_to_collection('checkpoints', feature_fastrcnn)
      fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss = self._create_losses(
        featuremap_labels, featuremap_boxes, fm_anchors, fm_shape, fastrcnn_box_logits, fastrcnn_label_logits,
        feature_fastrcnn, fg_inds_wrt_gt, gt_boxes, gt_masks, rcnn_labels, rcnn_sampled_boxes, rpn_box_logits,
        rpn_label_logits)
      final_boxes, final_labels, final_masks, final_probs = None, None, None, None
    else:
      final_boxes, final_labels, final_masks, final_probs = self._create_final_outputs(rpn_input,
                                                                                       fastrcnn_box_logits,
                                                                                       fastrcnn_label_logits,
                                                                                       proposal_boxes)
      fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss = None, None, None, None, None

    return final_boxes, final_labels, final_masks, final_probs, \
           fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss

  def _create_fastrcnn_output(self, boxes_on_featuremap, rpn_input):
    from datasets.COCO.COCO_detection import NUM_CLASSES
    roi_resized = roi_align(rpn_input, boxes_on_featuremap, 14)

    def ff_true():
      feature_fastrcnn = add_resnet_conv5(roi_resized, self.tower_setup)[0]
      fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_head(feature_fastrcnn, NUM_CLASSES, self.tower_setup)
      return feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits

    def ff_false():
      ncls = NUM_CLASSES
      return tf.zeros([0, 7, 7, 2048]), tf.zeros([0, ncls]), tf.zeros([0, ncls - 1, 4])

    feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits = tf.cond(
      tf.size(boxes_on_featuremap) > 0, ff_true, ff_false)
    return fastrcnn_box_logits, fastrcnn_label_logits, feature_fastrcnn

  def _create_losses(self, featuremap_labels, featuremap_boxes, fm_anchors, fm_shape, fastrcnn_box_logits,
                     fastrcnn_label_logits, feature_fastrcnn, fg_inds_wrt_gt, gt_boxes, gt_masks, rcnn_labels,
                     rcnn_sampled_boxes, rpn_box_logits, rpn_label_logits):
    anchor_labels = tf.slice(
      featuremap_labels, [0, 0, 0],
      tf.stack([fm_shape[0], fm_shape[1], -1]),
      name='sliced_anchor_labels')
    anchor_boxes = tf.slice(
      featuremap_boxes, [0, 0, 0, 0],
      tf.stack([fm_shape[0], fm_shape[1], -1, -1]),
      name='sliced_anchor_boxes')
    anchor_boxes_encoded = encode_bbox_target(anchor_boxes, fm_anchors)


    # rpn loss
    rpn_label_loss, rpn_box_loss = rpn_losses(
      anchor_labels, anchor_boxes_encoded, rpn_label_logits, rpn_box_logits)
    # fastrcnn loss
    fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])  # fg inds w.r.t all samples
    fg_sampled_boxes = tf.gather(rcnn_sampled_boxes, fg_inds_wrt_sample)
    matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)
    encoded_boxes = encode_bbox_target(
      matched_gt_boxes,
      fg_sampled_boxes) * tf.constant(FASTRCNN_BBOX_REG_WEIGHTS)
    fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_losses(
      rcnn_labels, fastrcnn_label_logits,
      encoded_boxes,
      tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample))
    if self._add_maskrcnn:
      # maskrcnn loss
      fg_labels = tf.gather(rcnn_labels, fg_inds_wrt_sample)
      fg_feature = tf.gather(feature_fastrcnn, fg_inds_wrt_sample)
      mask_logits = maskrcnn_head(fg_feature, NUM_CLASSES, self.tower_setup)  # #fg x #cat x 14x14

      gt_masks_for_fg = tf.gather(gt_masks, fg_inds_wrt_gt)  # nfg x H x W
      target_masks_for_fg = crop_and_resize(
        tf.expand_dims(gt_masks_for_fg, 3),
        fg_sampled_boxes,
        tf.range(tf.size(fg_inds_wrt_gt)), 14, pad_border=False)  # nfg x 1x14x14
      target_masks_for_fg = tf.squeeze(target_masks_for_fg, 3, 'sampled_fg_mask_targets')
      mrcnn_loss = maskrcnn_loss(mask_logits, fg_labels, target_masks_for_fg)
      self.add_scalar_summary(mrcnn_loss, "mrcnn_loss")
    else:
      mrcnn_loss = 0.0

    return fastrcnn_box_loss, fastrcnn_label_loss, mrcnn_loss, rpn_box_loss, rpn_label_loss

  def _create_final_outputs(self, rpn_input, fastrcnn_box_logits, fastrcnn_label_logits, proposal_boxes):
    label_probs = tf.nn.softmax(fastrcnn_label_logits, name='fastrcnn_all_probs')  # #proposal x #Class
    anchors = tf.tile(tf.expand_dims(proposal_boxes, 1), [1, NUM_CLASSES - 1, 1])  # #proposal x #Cat x 4
    decoded_boxes = decode_bbox_target(
      self.bbox_decode_clip,
      fastrcnn_box_logits /
      tf.constant(FASTRCNN_BBOX_REG_WEIGHTS), anchors)
    decoded_boxes = clip_boxes(decoded_boxes, self._image_shape2d, name='fastrcnn_all_boxes')
    # indices: Nx2. Each index into (#proposal, #category)
    pred_indices, final_probs = fastrcnn_predictions(decoded_boxes, label_probs)
    final_probs = tf.identity(final_probs, 'final_probs')
    final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
    final_labels = tf.add(pred_indices[:, 1], 1, name='final_labels')
    self.outputs = [final_probs, final_boxes, final_labels]
    if self._add_maskrcnn:
      # HACK to work around https://github.com/tensorflow/tensorflow/issues/14657
      def f1():
        roi_resized = roi_align(rpn_input, final_boxes * (1.0 / ANCHOR_STRIDE), 14)
        feature_maskrcnn = add_resnet_conv5(roi_resized, self.tower_setup)[0]
        mask_logits = maskrcnn_head(
          feature_maskrcnn, NUM_CLASSES, self.tower_setup)  # #result x #cat x 14x14
        indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
        mask_logits = tf.transpose(mask_logits, [0, 3, 1, 2])
        final_mask_logits = tf.gather_nd(mask_logits, indices)  # #resultx14x14
        return tf.sigmoid(final_mask_logits)

      final_masks = tf.cond(tf.size(final_probs) > 0, f1, lambda: tf.zeros([0, 14, 14]))
      final_masks = tf.identity(final_masks, name='final_masks')
      self.outputs.append(final_masks)
    else:
      final_masks = None
    return final_boxes, final_labels, final_masks, final_probs

  def _add_basic_measures(self, inp, loss):
    n_examples = tf.shape(inp)[0]
    self.measures[Measures.N_EXAMPLES] = n_examples
    if loss is not None:
      self.measures[Measures.LOSS] = loss * tf.cast(n_examples, tf.float32)

  def _add_test_measures(self, batch_idx, final_boxes, final_probs, final_labels, final_masks):
    if not self.is_training:
      orig_img_shape = self.network_input_dict[DataKeys.RAW_IMAGE_SIZES][batch_idx, ...]
      orig_img_shape_f = tf.cast(orig_img_shape, tf.float32)
      image_shape2d_f = tf.cast(self._image_shape2d, tf.float32)
      scale = (image_shape2d_f[0] / orig_img_shape_f[0] + image_shape2d_f[1] / orig_img_shape_f[1]) / 2
      boxes = final_boxes / scale
      self.measures[Measures.DET_BOXES] = clip_boxes(boxes, orig_img_shape_f)
      self.measures[Measures.DET_PROBS] = final_probs
      self.measures[Measures.DET_LABELS] = final_labels
      if self._add_maskrcnn:
        final_masks = tf.py_func(self.fill_full_mask, [self.measures[Measures.DET_BOXES], final_masks, orig_img_shape],
                                 tf.uint8, name="fill_full_mask")
        self.measures[Measures.DET_MASKS] = final_masks
      if DataKeys.IMAGE_ID in self.network_input_dict:
        self.measures[Measures.IMAGE_ID] = self.network_input_dict[DataKeys.IMAGE_ID]
