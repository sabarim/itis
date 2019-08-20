import glob
import numpy as np
from collections import namedtuple, OrderedDict


def load_tracking_gt_KITTI_format(gt_path, filter_to_cars=True, start_track_ids_from_1=True):
  tracking_gt = {}
  tracking_gt_files = glob.glob(gt_path + "*.txt")
  for tracking_gt_file in tracking_gt_files:
    seq = tracking_gt_file.split("/")[-1].replace(".txt", "")
    gt = np.genfromtxt(tracking_gt_file, dtype=np.str)
    if filter_to_cars:
      gt = gt[gt[:, 2] != "Cyclist"]
      gt = gt[gt[:, 2] != "Pedestrian"]
      gt = gt[gt[:, 2] != "Person"]
    gt = gt[gt[:, 2] != "DontCare"]
    if start_track_ids_from_1:
      # increase all track ids by 1 so we start at 1 instead of 0
      gt[:, 1] = (gt[:, 1].astype(np.int32) + 1).astype(np.str)

    tracking_gt[seq] = gt
  return tracking_gt


TrackingGtElement = namedtuple("TrackingGtElement", "time class_ id_ bbox_x0y0x1y1")


def load_tracking_gt_KITTI(gt_path, filter_to_cars_and_pedestrians):
  gt = load_tracking_gt_KITTI_format(gt_path, filter_to_cars=False, start_track_ids_from_1=False)
  if filter_to_cars_and_pedestrians:
    gt_filtered = {}
    for seq, gt_seq in gt.items():
      gt_seq = gt_seq[gt_seq[:, 2] != "Cyclist"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Van"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Person"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Tram"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Truck"]
      gt_seq = gt_seq[gt_seq[:, 2] != "Misc"]
      gt_filtered[seq] = gt_seq
    gt = gt_filtered
  # convert to TrackingGTElements which are much easier to use
  # and store it in nested dicts: seq -> id -> time -> TrackingGtElement
  nice_gt = {}
  for seq, seq_gt in gt.items():
    id_to_time_to_elem = OrderedDict()
    for gt_elem in seq_gt:
      t = int(gt_elem[0])
      id_ = int(gt_elem[1])
      class_ = gt_elem[2]
      bbox_x0y0x1y1 = gt_elem[6:10].astype("float")
      elem = TrackingGtElement(time=t, class_=class_, id_=id_, bbox_x0y0x1y1=bbox_x0y0x1y1)
      time_to_elem = id_to_time_to_elem.setdefault(id_, OrderedDict())
      time_to_elem[t] = elem
    nice_gt[seq] = id_to_time_to_elem
  return nice_gt
