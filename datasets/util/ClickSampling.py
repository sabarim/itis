import random
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import morphology


def get_sampled_locations(sample_locations, img_area, num_clicks, D=40):
  d_step = int(D / 2)
  img = np.copy(img_area)
  pts = []
  for click in range(num_clicks[0]):
    pixel_samples = list(zip(sample_locations[0], sample_locations[1]))
    if len(pixel_samples) > 1:
      [x, y] = random.sample(pixel_samples, 1)[0]
      pts.append([x, y])

      x_min = max(0, x - d_step)
      x_max = min(x + d_step, img.shape[0])
      y_min = max(0, y - d_step)
      y_max = min(y + d_step, img.shape[1])
      img[x_min:x_max, y_min:y_max] = 0

      sample_locations = np.where(img == 1)

  return pts


def get_image_area_to_sample(img, d_margin=5, D=40):
  """
  calculate set g_c, which has two properties
  1) They represent background pixels 
  2) They are within a certain distance to the object
  :param img: Image that represents the object instance
  """

  # TODO: In the paper 'Deep Interactive Object Selection', they calculate g_c first based on the original object instead
  # of the dilated one.

  # Dilate the object by d_margin pixels to extend the object boundary
  img_area = np.copy(img)
  # Ignore void label for dilation
  img_area[img != 1] = 0
  img_area = morphology.binary_dilation(img_area, morphology.diamond(d_margin)).astype(np.uint8)

  g_c = np.logical_not(img_area).astype(int)
  g_c[distance_transform_edt(g_c) > D] = 0
  g_c[img == 255] = 0

  return g_c