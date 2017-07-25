import numpy as np
import random

def extract_section(im, x, y, z, padding, section_shape):
    x_sect, y_sect, z_sect = section_shape
    size_x, size_y, size_z = im.shape
    take_x = x_sect + padding
    take_y = y_sect + padding
    take_z = z_sect + padding

    if x - (take_x / 2) < 0:
        sl_x = slice(0, take_x)
    elif x + (take_x / 2) > size_x:
        sl_x = slice(size_x - take_x, size_x)
    else:
        sl_x = slice(x - (take_x / 2), x + (take_x / 2))

    if y - (take_y / 2) < 0:
        sl_y = slice(0, take_y)
    elif y + (take_y / 2) > size_y:
        sl_y = slice(size_y - take_y, size_y)
    else:
        sl_y = slice(y - (take_y / 2), y + (take_y / 2))

    if z - (take_z / 2) < 0:
        sl_z = slice(0, take_z)
    elif z + (take_z / 2) > size_z:
        sl_z = slice(size_z - take_z, size_z)
    else:
        sl_z = slice(z - (take_z / 2), z + (take_z / 2))

    return im[sl_x, sl_y, sl_z]

def random_brain_points(gt):
    whole = np.array(np.where(gt != 0)).T
    core = np.array(np.where((gt != 0) & (gt != 2))).T
    active = np.array(np.where(gt == 4)).T
    healthy = np.array(np.where(gt == 0)).T

    region_candidates = [whole, core, active, healthy, healthy]
    regions = []
    for candidate in region_candidates:
        if len(candidate) > 0:
            regions.append(candidate)

    if len(regions) == 0:
        raise ValueError('Ground truth does not make sense.')

    reg = random.choice(regions)
    i = np.random.randint(0, len(reg))
    point = reg[i]
    return point

def random_hand_points(gt):
    bone = np.array(np.where(gt > 0)).T
    center = np.array([int(i) for i in np.round(np.mean(bone, axis=0))])
    offset = np.random.randint(-25, 25, size=(3,))
    point = center + offset

    # region_candidates = [metacarpal, proximal, middle]
    # regions = []
    # for candidate in region_candidates:
    #     if len(candidate) > 0:
    #         regions.append(candidate)
    #
    # if len(regions) == 0:
    #     raise ValueError('Ground truth does not make sense.')
    #
    # reg = random.choice(regions)
    # i = np.random.randint(0, len(reg))
    # point = reg[i]

    return point

def extract_random_section(image, gt, random_point_selection, section_shape):
    point = random_point_selection(gt.argmax(axis=0))
    x, y, z = point
    sections = [extract_section(modality, x, y, z, padding=0, section_shape=section_shape) for modality in image]
    gt_sections = [extract_section(class_map, x, y, z, padding=0, section_shape=section_shape) for class_map in gt]

    return np.array(sections, dtype='int16'), np.array(gt_sections, dtype='int8')


class RandomSectionSelection(object):
    def __init__(self, data_mode, followed_by=None):
        if data_mode == 'brain':
            self.random_point_selection = random_brain_points
            self.section_shape = (80, 72, 64)
        elif data_mode == 'hand':
            self.random_point_selection = random_hand_points
            self.section_shape = (144, 120, 96)
        elif data_mode == 'debug_hand':
            self.random_point_selection = random_hand_points
            self.section_shape = (64, 64, 64)
        elif data_mode == 'debug_brain':
            self.random_point_selection = random_brain_points
            self.section_shape = (32, 32, 32)
        else:
            raise ValueError('Data modes are: hand, brain.')
        self.data_mode = data_mode
        self.followed_by = followed_by

        self.__name__ = 'RandomSectionSelection'

    def __call__(self, x, z):
        n_classes = z.shape[-1]

        nx = np.transpose(x, (0, 2, 3, 4, 1))
        nz = np.reshape(z, (1, x.shape[3], x.shape[4], x.shape[1], n_classes))
        nz = np.transpose(nz, (0, 4, 1, 2, 3))

        nx, nz = extract_random_section(nx[0], nz[0], self.random_point_selection, self.section_shape)
        nx = np.transpose(nx[np.newaxis], (0, 4, 1, 2, 3))
        nz = np.transpose(nz[np.newaxis], (0, 2, 3, 4, 1))
        nz = np.reshape(nz, (nz.shape[0], nz.shape[1]*nz.shape[2]*nz.shape[3], n_classes))

        if self.followed_by is None:
            return (nx, nz)
        else:
            return self.followed_by(nx, nz)

def random_flip(x, z):
    flip_axis = random.choice([0, 1, 2])
    original_shape = z.shape
    nz = np.reshape(z, (1, x.shape[3], x.shape[4], x.shape[1], z.shape[-1]))
    if flip_axis == 0:
        # flip along the height
        nx = x[:, :, :, ::-1, :]
        nz = nz[:, ::-1, :, :, :]
    elif flip_axis == 1:
        # flip along the width
        nx = x[:, :, :, :, ::-1]
        nz = nz[:, :, ::-1, :, :]
    else:
        # flip along the depth
        nx = x[:, ::-1, :, :, :]
        nz = nz[:, :, :, ::-1, :]
    nz = np.reshape(nz, original_shape)
    return (nx, nz)

def percentile_filter(x, z):
    from scipy.ndimage import percentile_filter
    from breze.learn.data import one_hot
    percentile = np.random.randint(0, 10)

    nx = np.transpose(x, (0, 2, 1, 3, 4))
    nx[0] = [percentile_filter(modality, percentile, (2, 2, 2)) for modality in nx[0]]
    nx = np.transpose(nx, (0, 2, 1, 3, 4))

    n_classes = z.shape[-1]
    nz = np.reshape(z, (x.shape[3], x.shape[4], x.shape[1], n_classes))
    nz = np.transpose(nz, (3, 0, 1, 2))
    nz = np.array([percentile_filter(class_map, percentile, (2, 2, 2)) for class_map in nz])
    nz = nz.argmax(axis=0)
    nz = np.reshape(nz, (-1,))
    nz = np.reshape(one_hot(nz, n_classes), z.shape)

    nx = np.asarray(nx, dtype=x.dtype)
    nz = np.asarray(nz, dtype=z.dtype)

    return (nx, nz)

def swirl_(im, strength, radius):
    from skimage.transform import swirl
    return [swirl(im_slice, rotation=0, strength=strength, radius=radius) for im_slice in im]

def swirl_transform(x, z):
    """
    Adds a swirl effect to every depth slice.
    Assuming a batch size of 1.
    More specifically: x is (1, depth, channels, height, width) and z is (1, height*width*depth, classes)
    """
    from breze.learn.data import one_hot
    strength = np.random.uniform(1, 2)
    radius = np.random.randint(90, 140)
    z_original_shape = z.shape
    n_classes = z.shape[-1]

    nx = np.transpose(x, (0, 2, 1, 3, 4))
    nz = np.reshape(z, (1, x.shape[3], x.shape[4], x.shape[1], n_classes))
    nz = np.transpose(nz, (0, 4, 3, 1, 2))
    nx[0] = [swirl_(modality, strength, radius) for modality in nx[0]]
    nx = np.transpose(nx, (0, 2, 1, 3, 4))
    nz[0] = [swirl_(class_map, strength, radius) for class_map in nz[0]]
    nz = nz[0].argmax(axis=0)
    nz = np.transpose(nz, (1, 2, 0))
    nz = np.reshape(nz, (-1,))
    nz = np.reshape(one_hot(nz, n_classes), z_original_shape)

    nx = np.asarray(nx, dtype=x.dtype)
    nz = np.asarray(nz, dtype=z.dtype)

    return (nx, nz)

def minor_rotation(x, z):
    """
    Assuming a batch size of 1.
    More specifically: x is (1, depth, channels, height, width) and z is (1, height*width*depth, classes)
    """
    from scipy.ndimage.interpolation import rotate as rotate_scipy
    from breze.learn.data import one_hot
    z_original_shape = z.shape
    n_classes = z.shape[-1]
    ang = float(np.random.uniform(-90, 90))
    axes = np.random.permutation(3)[:2]

    nx = np.transpose(x, (0, 2, 3, 4, 1))
    nz = np.reshape(z, (1, x.shape[3], x.shape[4], x.shape[1], n_classes))
    nz = np.transpose(nz, (0, 4, 1, 2, 3))

    nx[0] = [rotate_scipy(modality, ang, axes=axes, order=3, reshape=False) for modality in nx[0]]
    nx = np.transpose(nx, (0, 4, 1, 2, 3))
    nz[0] = [rotate_scipy(class_map, ang, axes=axes, order=3, reshape=False) for class_map in nz[0]]
    nz = nz[0].argmax(axis=0)
    nz = np.reshape(nz, (-1,))
    nz = np.reshape(one_hot(nz, n_classes), z_original_shape)

    nx = np.asarray(nx, dtype=x.dtype)
    nz = np.asarray(nz, dtype=z.dtype)

    return (nx, nz)

def full_rotation(x, z):
    """
    Assuming a batch size of 1.
    More specifically: x is (1, depth, channels, height, width) and z is (1, height*width*depth, classes)
    """
    from scipy.ndimage.interpolation import rotate as rotate_scipy
    from breze.learn.data import one_hot
    z_original_shape = z.shape
    n_classes = z.shape[-1]
    ang = float(np.random.uniform(0, 360))
    axes = np.random.permutation(3)[:2]

    nx = np.transpose(x, (0, 2, 3, 4, 1))
    nz = np.reshape(z, (1, x.shape[3], x.shape[4], x.shape[1], n_classes))
    nz = np.transpose(nz, (0, 4, 1, 2, 3))

    nx[0] = [rotate_scipy(modality, ang, axes=axes, order=3, reshape=False) for modality in nx[0]]
    nx = np.transpose(nx, (0, 4, 1, 2, 3))
    nz[0] = [rotate_scipy(class_map, ang, axes=axes, order=3, reshape=False) for class_map in nz[0]]
    nz = nz[0].argmax(axis=0)
    nz = np.reshape(nz, (-1,))
    nz = np.reshape(one_hot(nz, n_classes), z_original_shape)

    nx = np.asarray(nx, dtype=x.dtype)
    nz = np.asarray(nz, dtype=z.dtype)

    return (nx, nz)

def identity(x, z):
    return (x, z)

def nil(x, z):
    nx = np.zeros(x.shape)
    nz = np.zeros(z.shape)

    return (nx, nz)

def random_transformation(x, z):
    import random
    transformations = ['identity', 'random_flip', 'percentile_filter', 'full_rotation']
    transform_dict = {
        'identity': identity,
        'random_flip': random_flip,
        'percentile_filter': percentile_filter,
        'full_rotation': full_rotation
    }

    transform_key = random.choice(transformations)
    transform_fun = transform_dict[transform_key]

    nx, nz = transform_fun(x, z)

    second_transform_key = random.choice(transformations)
    if second_transform_key == transform_key:
        return (nx, nz)
    else:
        second_transform_fun = transform_dict[second_transform_key]
        return second_transform_fun(nx, nz)

def random_geometric_transformation(x, z):
    import random
    transformations = ['identity', 'random_flip', 'full_rotation']
    transform_dict = {
        'identity': identity,
        'random_flip': random_flip,
        'full_rotation': full_rotation
    }

    transform_key = random.choice(transformations)
    transform_fun = transform_dict[transform_key]

    nx, nz = transform_fun(x, z)

    second_transform_key = random.choice(transformations)
    if second_transform_key == transform_key:
        return (nx, nz)
    else:
        second_transform_fun = transform_dict[second_transform_key]
        return second_transform_fun(nx, nz)

def random_soft_geometric_transformation(x, z):
    import random
    transformations = ['identity', 'random_flip', 'full_rotation']
    transform_dict = {
        'identity': identity,
        'random_flip': random_flip,
        'full_rotation': full_rotation
    }

    transform_key = random.choice(transformations)
    transform_fun = transform_dict[transform_key]

    nx, nz = transform_fun(x, z)

    return (nx, nz)