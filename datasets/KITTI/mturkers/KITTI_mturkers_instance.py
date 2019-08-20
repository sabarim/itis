import glob

from datasets.Dataset import FileListDataset
from datasets.Loader import register_dataset
from datasets.util.Util import username

NAME="kitti_mturkers_instance"
DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/KITTI/"

@register_dataset(NAME)
class KITTIMturkersInstanceDataset(FileListDataset):
  def __init__(self, config, subset, name = NAME):
    super(KITTIMturkersInstanceDataset, self).__init__(config,name, subset, DEFAULT_PATH, 2)

  def read_inputfile_lists(self):
    files = glob.glob(self.data_dir + "object/segmentations_jay_per_instance_new/*.png")
    imgs = [self.data_dir + "object/image_2/" + file.split("/")[-1].split(":")[0] + ".png" for file in files]

    return imgs, files