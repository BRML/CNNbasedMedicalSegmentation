import cPickle as pickle
import json
import os
import time
import sys
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
from skimage import color
from breze.learn.data import one_hot
import matplotlib.pyplot as plt
import numpy as np
import h5py
import gnumpy

import climin.stops
from climin import mathadapt as ma

import ash
from model_defs import get_model

from conv3d.model import SequentialModel

def to_sections(im):
    check = [i % 2 == 0 for i in im.shape]
    if not all(check):
        raise ValueError('All dimensions must be even numbers. Got: %s' % check)
    size_x, size_y, size_z = im.shape
    step_x = size_x / 2
    step_y = size_y / 2
    step_z = size_z / 2

    sections = []
    for z in range(2):
        for y in range(2):
            for x in range(2):
                begin_x = step_x*x
                begin_y = step_y*y
                begin_z = step_z*z

                end_x = begin_x + step_x
                end_y = begin_y + step_y
                end_z = begin_z + step_z

                section = im[begin_x:end_x, begin_y:end_y, begin_z:end_z].copy()
                sections.append(section)
    return sections

def from_sections(sections, original_shape):
    if len(sections) == 0:
        raise ValueError('Section list is empty.')
    im = np.zeros(original_shape, dtype=sections[0].dtype)

    check = [i % 2 == 0 for i in im.shape]
    if not all(check):
        raise ValueError('All dimensions must be even numbers. Got: %s' % check)
    size_x, size_y, size_z = im.shape
    step_x = size_x / 2
    step_y = size_y / 2
    step_z = size_z / 2

    count = 0
    for z in range(2):
        for y in range(2):
            for x in range(2):
                begin_x = step_x*x
                begin_y = step_y*y
                begin_z = step_z*z

                end_x = begin_x + step_x
                end_y = begin_y + step_y
                end_z = begin_z + step_z

                section = sections[count]
                count += 1
                im[begin_x:end_x, begin_y:end_y, begin_z:end_z] = section[:,:,:].copy()

    return im

def dice_alt(seg, gt):
    sim = float(np.sum(np.minimum(seg, gt)))
    sim /= np.sum(np.maximum(seg, gt))
    return sim

def dice_(seg, gt):
    intersection = 2. * np.sum(seg * gt)
    denominator = (np.sum(np.square(seg)) + np.sum(np.square(gt)))
    if denominator == 0:
        return 1.
    similarity = intersection / denominator
    return similarity

def dice(seg, gt):
    seg_transposed = np.transpose(seg, (3, 0, 1, 2))
    gt = np.transpose(gt, (3, 0, 1, 2))

    dice_list = [dice_(s, g) for s, g in zip(seg_transposed, gt)]
    return dice_list

def get_whole(map):
    healthy = map[:, :, :, 0]
    non_healthy = np.sum(map[:, :, :, 1:], axis=3)

    result = np.zeros((map.shape[:3] + (2,)))
    result[:, :, :, 0] = healthy
    result[:, :, :, 1] = non_healthy

    return result.argmax(axis=3)

def get_core(map):
    core = map[:, :, :, 1] + np.sum(map[:, :, :, 3:], axis=3)
    non_core = map[:, :, :, 0] + map[:, :, :, 2]

    result = np.zeros((map.shape[:3] + (2,)))
    result[:, :, :, 0] = non_core
    result[:, :, :, 1] = core

    return result.argmax(axis=3)

def get_active(map):
    active = map[:, :, :, 3]
    non_active = np.sum(map[:, :, :, :3], axis=3) + map[:, :, :, 4]

    result = np.zeros((map.shape[:3] + (2,)))
    result[:, :, :, 0] = non_active
    result[:, :, :, 1] = active

    return result.argmax(axis=3)

def brats_dice(seg, gt):
    whole_seg = get_whole(seg)
    whole_gt = get_whole(gt)
    core_seg = get_core(seg)
    core_gt = get_core(gt)
    active_seg = get_active(seg)
    active_gt = get_active(gt)

    seg_and_gt = [(whole_seg, whole_gt), (core_seg, core_gt), (active_seg, active_gt)]
    dice_list = [dice_(s, g) for s, g in seg_and_gt]

    return dice_list

def discrete(seg, n_classes):
    original_shape = seg.shape
    discrete_seg = seg.argmax(axis=3)
    discrete_seg = np.reshape(discrete_seg, (-1,))
    discrete_seg = np.reshape(one_hot(discrete_seg, n_classes), original_shape)

    return discrete_seg

def vis_col_result(im, seg, gt, savefile=None):
    indices_0 = np.where(gt == 0)
    indices_1 = np.where(gt == 1)  # metacarpal
    indices_2 = np.where(gt == 2)  # proximal
    indices_3 = np.where(gt == 3)  # middle (thumb: distal)
    indices_4 = np.where(gt == 4)  # distal (thumb: none)

    indices_s0 = np.where(seg == 0)
    indices_s1 = np.where(seg == 1)
    indices_s2 = np.where(seg == 2)
    indices_s3 = np.where(seg == 3)
    indices_s4 = np.where(seg == 4)

    im = im * 1. / im.max()
    rgb_image = color.gray2rgb(im)
    m0 = [0.6, 0.6, 1.]
    m1 = [0.2, 1., 0.2]
    m2 = [1., 1., 0.2]
    m3 = [1., 0.6, 0.2]
    m4 = [1., 0., 0.]

    im_gt = rgb_image.copy()
    im_seg = rgb_image.copy()
    im_gt[indices_0[0], indices_0[1], :] *= m0
    im_gt[indices_1[0], indices_1[1], :] *= m1
    im_gt[indices_2[0], indices_2[1], :] *= m2
    im_gt[indices_3[0], indices_3[1], :] *= m3
    im_gt[indices_4[0], indices_4[1], :] *= m4

    im_seg[indices_s0[0], indices_s0[1], :] *= m0
    im_seg[indices_s1[0], indices_s1[1], :] *= m1
    im_seg[indices_s2[0], indices_s2[1], :] *= m2
    im_seg[indices_s3[0], indices_s3[1], :] *= m3
    im_seg[indices_s4[0], indices_s4[1], :] *= m4

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(im_seg)
    a.set_title('Segmentation')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(im_gt)
    a.set_title('Ground truth')
    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.show()
    plt.close()

def vis_result(image, seg, gt, title1='Segmentation', title2='Ground truth', savefile=None):
    indices = np.where(seg >= 0.5)
    indices_gt = np.where(gt >= 0.5)

    im_norm = image / image.max()
    rgb_image = color.gray2rgb(im_norm)
    multiplier = [0., 1., 1.]
    multiplier_gt = [1., 1., 0.]

    im_seg = rgb_image.copy()
    im_gt = rgb_image.copy()
    im_seg[indices[0], indices[1], :] *= multiplier
    im_gt[indices_gt[0], indices_gt[1], :] *= multiplier_gt

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(im_seg)
    a.set_title(title1)
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(im_gt)
    a.set_title(title2)

    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)
    plt.close()

def build_net(model_folder, model_code, n_classes, train_size, inpt_h, inpt_w, inpt_d, n_channels):
    model_path = os.path.join('models', model_folder)

    param_file = os.path.join(model_path, 'params.hdf5')
    bn_par_file = os.path.join(model_path, 'bn_pars.pkl')

    log = None
    for f_name in os.listdir(model_path):
        if f_name.endswith('.json') and not f_name.startswith('dice'):
            with open(os.path.join(model_path, f_name), 'r') as f:
                log = json.load(f)
            break
    if 'layers' not in log:
        log = None

    model_def = get_model(model_code)

    layer_vars = model_def.layer_vars if log is None else log['layers']
    batchnorm = model_def.batchnorm
    loss_id = model_def.loss_id
    out_transfer = model_def.out_transfer

    batch_size = 1
    max_passes = 1
    inpt_dims = (inpt_h, inpt_w, inpt_d)

    n_report = train_size / batch_size
    max_iter = n_report * max_passes

    optimizer = 'adam'

    model = SequentialModel(
        image_height=inpt_dims[0], image_width=inpt_dims[1],
        image_depth=inpt_dims[2], n_channels=n_channels,
        n_output=n_classes, layer_vars=layer_vars,
        out_transfer=out_transfer, loss_id=loss_id,
        optimizer=optimizer, batch_size=batch_size,
        max_iter=max_iter, using_bn=batchnorm
    )

    f_params = h5py.File(param_file, 'r')
    params = np.zeros(model.parameters.data.shape)
    params[...] = f_params['best_pars']
    f_params.close()
    model.parameters.data[...] = params

    if batchnorm and os.path.exists(bn_par_file):
        with open(bn_par_file, 'r') as f:
            bn_pars = pickle.load(f)
            model.set_batchnorm_params(bn_pars)
    else:
        if batchnorm:
            raise AssertionError('Batch norm used but running metrics not available.')

    if batchnorm:
        predict = ash.BatchNormFuns(
            model=model,
            fun=model.predict,
            phase='infer'
        )
    else:
        predict = model.predict

    return predict

def get_data(data_name, x_only=False):
    data_path = os.path.join('data', 'datasets', data_name+'.hdf5')
    data = h5py.File(data_path, 'r')

    test_x = data['test_x']
    test_y = data['test_y'] if not x_only else None

    return test_x, test_y

def compute_results(predict, X, Y):
    for x, y in zip(X, Y):
        depth, n_channels, height, width = x.shape
        start = time.time()
        model_output = predict(x[np.newaxis])
        end = time.time()
        n_classes = y.shape[-1]
        model_output = model_output.as_numpy_array() if isinstance(model_output, gnumpy.garray) else model_output
        seg = np.reshape(
            model_output,
            (height, width, depth, n_classes)
        )
        gt = np.reshape(y, (height, width, depth, n_classes))
        seg = discrete(seg, n_classes)
        dice_list = dice(seg, gt)
        seg = seg.argmax(axis=3)
        gt = gt.argmax(axis=3)
        dice_all = dice_alt(seg, gt)
        dice_list = [dice_all] + dice_list
        print '\tdice: ', dice_list
        print '\ttime taken: ', (end - start)
        print '-' * 20

        image = x[:, 0, :, :]
        image = np.transpose(image, (1, 2, 0))
        yield (image, seg, gt, dice_list)

class SeqPredict(object):
    def __init__(self, predict, n_classes):
        self.predict = predict
        self.n_classes = n_classes

    def __call__(self, x):
        n_classes = self.n_classes
        image = np.transpose(x[0], (1, 2, 3, 0))
        sections = np.array([to_sections(modality) for modality in image], dtype='int16') # mod sect h w d
        sections = np.transpose(sections, (1, 4, 0, 2, 3))
        seg_sections = []
        for section in sections:
            depth, n_chans, height, width = section.shape
            model_output = self.predict(section[np.newaxis])
            model_output = model_output.as_numpy_array() if isinstance(model_output, gnumpy.garray) else model_output
            seg = np.reshape(
                model_output,
                (height, width, depth, n_classes)
            )
            seg = seg.argmax(axis=3)
            seg_sections.append(seg)
        final_seg = from_sections(seg_sections, original_shape=(x.shape[3], x.shape[4], x.shape[1]))

        seg_onehot = np.reshape(final_seg, (-1,))
        seg_onehot = np.reshape(one_hot(seg_onehot, n_classes), (-1, n_classes))

        return seg_onehot

def save_results(image, seg, gt, result_path):
    slice_count = 0
    for _slice in np.arange(0, image.shape[-1], 1):
        im_slice = image[:, :, _slice]
        gt_slice = gt[:, :, _slice]
        seg_slice = seg[:, :, _slice]

        save_file = os.path.join(result_path, 'slice' + str(slice_count) + '.png')
        vis_col_result(im=im_slice, gt=gt_slice, seg=seg_slice, savefile=save_file)
        slice_count += 1

def test():
    model_folder = os.path.join('brats_fold0', 'session1_2')
    model_code = 'fcn_rffc2'
    data_name = 'brats2013_leaderboard_data'
    save_path = os.path.join('results', 'as_hdf', 'brats2013_leaderboard_results.hdf5')

    tx, _ = get_data(data_name, x_only=True)

    print 'Saving results to: ', save_path
    save_hdf5 = h5py.File(save_path, 'w')
    seg_maps = save_hdf5.create_dataset(
        'test_result', (tx.shape[0], tx.shape[3], tx.shape[4], tx.shape[1]), dtype='int8')

    train_size, depth, n_chans, height, width = tx.shape
    n_classes = 5
    predict = build_net(model_folder, model_code,
                        n_classes=n_classes, train_size=train_size,
                        inpt_h=height, inpt_w=width, inpt_d=depth,
                        n_channels=n_chans)

    index = 0
    for test_image in tx:
        model_output = predict(test_image[np.newaxis])
        model_output = model_output.as_numpy_array() if isinstance(model_output, gnumpy.garray) else model_output
        fuzzy_seg = np.reshape(
            model_output,
            (height, width, depth, n_classes)
        )
        seg = fuzzy_seg.argmax(axis=3)
        seg_maps[index,:,:,:] = seg

        index += 1

    save_hdf5.close()

def demonstrate():
    model_folder = os.path.join('brats_fold0', 'session1_2')
    model_code = 'fcn_rffc4'
    data_name = 'brats_fold0'
    sectionalized = False

    tx, ty = get_data(data_name)
    train_size, depth, n_chans, height, width = tx.shape
    n_classes = ty.shape[-1]
    if sectionalized:
        depth = depth / 2
        height = height / 2
        width = width / 2
    predict = build_net(model_folder, model_code,
                        n_classes=n_classes, train_size=train_size,
                        inpt_h=height, inpt_w=width, inpt_d=depth,
                        n_channels=n_chans)

    if sectionalized:
        predict = SeqPredict(predict=predict, n_classes=n_classes)

    count = 1
    dice_lists = []
    for image, seg, gt, dl in compute_results(predict, tx, ty):
        dice_lists.append(dl)
        result_path = os.path.join('results', model_folder, 'testim_'+str(count))
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        save_results(image, seg, gt, result_path)

        count += 1
    dice_matrix = np.array(dice_lists)
    dice_means = np.mean(dice_matrix, axis=0)
    print 'Mean dice values: ', dice_means

if __name__ == '__main__':
    demonstrate()
    #test()