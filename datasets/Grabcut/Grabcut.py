import glob
import tensorflow as tf

from datasets.Dataset import FileListDataset
from datasets.Loader import register_dataset
from datasets.util.Util import username

NAME = "grabcut"
DEFAULT_PATH="/fastwork/" + username() + "/mywork/data/Grabcut/"

@register_dataset(NAME)
class GrabcutDataset(FileListDataset):
  def __init__(self, config, subset, name=NAME):
    super().__init__(config, dataset_name=name, subset=subset, default_path=DEFAULT_PATH, num_classes=2)

  def postproc_annotation(self, ann_filename, ann):
    ann_postproc = tf.where(tf.equal(ann, 255), tf.ones_like(ann), ann)
    ann_postproc = tf.where(tf.equal(ann_postproc, 128), tf.ones_like(ann) * 255, ann_postproc)
    return ann_postproc

  def read_inputfile_lists(self):
    img_dir = self.data_dir + "images/"
    gt_dir = self.data_dir + "images-gt/"
    imgs = []
    gts = []

    for filename in glob.glob(img_dir + "*"):
      fn_base = filename.split("/")[-1].rsplit(".")[0]
      imgs += [filename]
      gts += [gt_dir + fn_base + ".png"]

    return imgs, gts