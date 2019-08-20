from datasets import DataKeys
from datasets.Dataset import FileListDataset
from datasets.KITTI.mturkers.KITTI_mturkers_instance import KITTIMturkersInstanceDataset
from datasets.Loader import register_dataset

NAME = "kitti_masked_dios"


@register_dataset(NAME)
class KITTIMaskedDiosDataset(KITTIMturkersInstanceDataset):
  def __init__(self, config, subset, name=NAME):
    super().__init__(config, subset, name)

  def get_extraction_keys(self):
    return self.pascal_masked_dataset.get_extraction_keys()

  def postproc_example_before_assembly(self, tensors):
    return self.pascal_masked_dataset.postproc_example_before_assembly(tensors)

  def use_segmentation_mask(self, res):
    self.pascal_masked_dataset.use_segmentation_mask(res)

  def postproc_annotation(self, ann_filename, ann):
    mask = super().postproc_annotation(ann_filename, ann)
    return {DataKeys.SEGMENTATION_LABELS: mask, DataKeys.RAW_SEGMENTATION_LABELS: mask,
            DataKeys.IMAGE_FILENAMES: ann_filename}