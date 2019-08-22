from datasets.Dataset import FileListDataset
from datasets.util.Util import username
from datasets.Loader import register_dataset

# DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/PascalVOC/benchmark_RELEASE/dataset/"
DEFAULT_PATH = "data/"
NAME = "pascalvoc"


@register_dataset(NAME)
class PascalVOCDataset(FileListDataset):
  def __init__(self, config, name, subset, num_classes):
    data_dir = config.string("data_dir", DEFAULT_PATH)
    super().__init__(config, name, subset, data_dir, num_classes)

  def read_inputfile_lists(self):
    data_list = "train.txt" if self.subset == "train" else "val.txt"
    data_list = "datasets/PascalVOC/" + data_list
    imgs = []
    ans = []
    with open(data_list) as f:
      for l in f:
        im, an = l.strip().split()
        im = self.data_dir + im
        an = self.data_dir + an
        imgs.append(im)
        ans.append(an)
    return imgs, ans
