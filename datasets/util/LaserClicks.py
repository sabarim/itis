import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_erosion, binary_dilation


def create_random_laser_clicks(mask, laser_min_spacing, laser_max_spacing, laser_grid_offset_stddev,
                               laser_min_keep_prob, laser_max_keep_prob, laser_n_max_rectangles,
                               laser_min_rectangle_size, laser_max_rectangle_size, laser_boundary_detect_size,
                               laser_boundary_extend_size, laser_swap_min_prob, laser_swap_max_prob,
                               put_gaussians=True):
  # note: mask can contain void (255)
  # map background to -1, keep foreground as 1 and set void to 0
  clicks = mask.astype(np.float32)
  clicks[clicks == 0] = -1
  clicks[clicks == 255] = 0

  # put a regular grid on the image
  laser_spacing_y = np.random.randint(laser_min_spacing, laser_max_spacing)
  laser_spacing_x = np.random.randint(laser_min_spacing, laser_max_spacing)
  col = np.arange(clicks.shape[0]) % laser_spacing_y == 0
  row = np.arange(clicks.shape[1]) % laser_spacing_x == 0
  grid = np.logical_and(col[:, np.newaxis], row[np.newaxis, :])

  # deform the grid
  coords_y, coords_x = grid.nonzero()

  random_offset_y = np.round(np.random.normal(scale=laser_grid_offset_stddev, size=coords_y.shape)).astype(np.int)
  coords_y += random_offset_y
  coords_y = np.minimum(coords_y, mask.shape[0] - 1)

  random_offset_x = np.round(np.random.normal(scale=laser_grid_offset_stddev, size=coords_y.shape)).astype(np.int)
  coords_x += random_offset_x
  coords_x = np.minimum(coords_x, mask.shape[1] - 1)
  grid_new = np.zeros_like(grid)
  grid_new[coords_y, coords_x] = 1

  # apply grid
  clicks *= grid_new[:, :, np.newaxis]

  # drop (uniformly) randomly
  p_keep = np.random.uniform(laser_min_keep_prob, laser_max_keep_prob)
  keep_mask = np.random.binomial(1, p_keep, clicks.shape)
  clicks *= keep_mask

  # cut out some random rectangles
  n_rectangles = np.random.randint(0, laser_n_max_rectangles)
  for _ in range(n_rectangles):
    y0 = np.random.randint(mask.shape[0])
    x0 = np.random.randint(mask.shape[1])
    h = np.random.randint(laser_min_rectangle_size, laser_max_rectangle_size)
    w = np.random.randint(laser_min_rectangle_size, laser_max_rectangle_size)
    clicks[y0:y0+h, x0:x0+w] = 0

  # swap some labels randomly around boundaries
  swap_labels_at_boundary(clicks, mask, laser_boundary_detect_size, laser_boundary_extend_size, laser_swap_min_prob,
                          laser_swap_max_prob)

  # place Gaussians by convolving with Gaussian
  if put_gaussians:
    clicks = put_gaussians_at_clicks(clicks)

  return clicks


def swap_labels_at_boundary(clicks, mask, boundary_detect_size=5, boundary_extend_size=25, swap_min_prob=0.2,
                            swap_max_prob=0.4):
  obj_mask = mask[:, :, 0] == 1
  eroded = binary_erosion(obj_mask, structure=np.ones((boundary_detect_size, boundary_detect_size)))
  boundary = np.logical_and(obj_mask, np.logical_not(eroded))
  boundary = binary_dilation(boundary, structure=np.ones((boundary_extend_size, boundary_extend_size)))
  # import matplotlib.pyplot as plt
  # plt.imshow(boundary)
  # plt.show()
  p_swap = np.random.uniform(swap_min_prob, swap_max_prob)
  swap_mask = np.random.binomial(1, p_swap, clicks.shape).astype(np.bool)
  swap_mask = np.logical_and(swap_mask, boundary[:, :, np.newaxis])
  clicks[swap_mask] *= -1


def put_labels_and_gaussians_on_laser(mask, laser):
  # note: mask can contain void (255)
  # map background to -1, keep foreground as 1 and set void to 0
  clicks = mask.astype(np.float32)
  clicks[clicks == 0] = -1
  clicks[clicks == 255] = 0

  # swap some labels randomly around boundaries
  swap_labels_at_boundary(clicks, mask)

  # restrict to sparse laser points coming from GAN
  clicks *= laser
  clicks = put_gaussians_at_clicks(clicks)
  return clicks


def put_gaussians_at_clicks(clicks):
  if clicks.ndim == 3:
    assert clicks.shape[-1] == 1
    clicks = clicks[:, :, 0]
  clicks = gaussian_filter(clicks, sigma=1.0, truncate=3.0) * 5
  clicks = clicks[:, :, np.newaxis]
  return clicks


def _test_laser_clicks():
  from PIL import Image
  mask = np.array(Image.open("/work/voigtlaender/data/LASER/KITTI/KITTI_laser_v1/000000.png"))
  img = np.array(Image.open("/work/voigtlaender/data/LASER/KITTI/KITTI_laser_v1/JPEGImages/480p/0000/000000.png"))
  # TODO: we should introduce some void pixels to test that as well
  img_raw = (img * 0.6).round().astype("uint8")
  for n in range(20):
    img = img_raw.copy()
    # TODO: add values for all the newly introduced params
    clicks = create_random_laser_clicks(mask[:, :, np.newaxis] / 255, put_gaussians=False)
    img[clicks[:, :, 0] == 1, 0] = 255
    img[clicks[:, :, 0] == -1, 1] = 255
    Image.fromarray(img).save("/work/voigtlaender/tmp/{}.png".format(n))
    import matplotlib.pyplot as plt
    #plt.imshow(img)
    #plt.show()


if __name__ == "__main__":
  _test_laser_clicks()
