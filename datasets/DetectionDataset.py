import tensorflow as tf
from functools import partial

from datasets.Dataset import FileListDataset
from datasets.util.Detection import init_anchors, add_rpn_data, load_instance_seg_annotation_np
from datasets import DataKeys


class DetectionFileListDataset(FileListDataset):
  def __init__(self, config, dataset_name, subset, default_path, num_classes, n_max_detections,
               class_ids_with_instances, id_divisor):
    super().__init__(config, dataset_name, subset, default_path, num_classes)
    self.add_masks = config.bool("add_masks", True)
    init_anchors(config)
    self._n_max_detections = n_max_detections
    self._class_ids_with_instances = class_ids_with_instances
    self._id_divisor = id_divisor

  def assemble_example(self, tensors):
    tensors = super().assemble_example(tensors)
    tensors = add_rpn_data(tensors)
    return tensors

  def load_annotation(self, img, img_filename, annotation_filename):
    load_ann_np = partial(load_instance_seg_annotation_np, n_max_detections=self._n_max_detections,
                          class_ids_with_instances=self._class_ids_with_instances, id_divisor=self._id_divisor)
    bboxes, ids, classes, is_crowd, mask = tf.py_func(load_ann_np, [annotation_filename],
                                                      [tf.float32, tf.int32, tf.int32, tf.int32, tf.uint8],
                                                      name="postproc_ann_np")
    bboxes.set_shape((self._n_max_detections, 4))
    ids.set_shape((self._n_max_detections,))
    classes.set_shape((self._n_max_detections,))
    is_crowd.set_shape((self._n_max_detections,))
    mask.set_shape((None, None, self._n_max_detections))

    return_dict = {DataKeys.BBOXES_y0x0y1x1: bboxes, DataKeys.CLASSES: classes, DataKeys.IDS: ids,
                   DataKeys.IS_CROWD: is_crowd}
    if self.add_masks:
      return_dict[DataKeys.SEGMENTATION_MASK] = mask
    return return_dict
