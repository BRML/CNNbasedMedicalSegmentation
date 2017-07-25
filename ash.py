import time
import sys
import exceptions
import json
import os
import cPickle as pickle

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import matplotlib.pyplot as plt
from skimage import color

import h5py
import gnumpy

from climin import mathadapt as ma
from climin.util import iter_minibatches

from breze.arch.util import lookup
from breze.arch.component import transfer as _transfer

def dice_demo_(seg, gt):
    dice = np.sum(2 * seg * gt)
    dice /= (np.sum(np.square(seg)) + np.sum(np.square(gt)))
    return dice


def dice_demo(seg, gt):
    seg_transposed = np.transpose(seg, (3, 0, 1, 2))
    gt = np.transpose(gt, (3, 0, 1, 2))

    dice_list = [dice_demo_(s, g) for s, g in zip(seg_transposed, gt)]
    dice_list = [dice_demo_(seg_transposed, gt)] + dice_list
    return dice_list

def vis_result(image, seg, gt, file_name='test.png'):
    indices = np.where(seg == 1)
    indices_gt = np.where(gt == 1)

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
    a.set_title('Segmentation')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(im_gt)
    a.set_title('Ground truth')
    plt.savefig(file_name)

def vis_col_im(im, seg, gt, file_name='test.png'):
    indices_0 = np.where(gt == 0)
    indices_1 = np.where(gt == 1)  # green - metacarpal - necrosis
    indices_2 = np.where(gt == 2)  # yellow - proximal - edema
    indices_3 = np.where(gt == 3)  # orange - middle - enhancing tumor
    indices_4 = np.where(gt == 4)  # red - distal - nonenhancing tumor

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
    plt.savefig(file_name)

    plt.close()

class TransFun(object):
    def __init__(self, fun, *params):
        self.params = params
        self.fun = fun

    def __call__(self, inpt):
        return self.fun(inpt, *self.params)

def tensor_softmax(inpt, n_classes=2):
    output = inpt.dimshuffle(0, 3, 4, 1, 2)
    output = T.reshape(output, (-1, n_classes))

    f = lookup('softmax', _transfer)
    output = T.reshape(f(output), (1, -1, n_classes))
    return output

def tensor_ident(inpt, n_classes=2):
    output = inpt.dimshuffle(0, 3, 4, 1, 2)
    output = T.reshape(output, (1, -1, n_classes))
    return output

def fcn_cat_ce(target, prediction, eps=1e-8):
    '''
    This loss function assumes the data set is processed one
    image (patch) at a time. As a consequence, the targets and
    the predictions should both be of shape (1, n_voxels, n_classes).
    '''
    prediction = T.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
    target = T.reshape(target, (target.shape[1], target.shape[2]))
    prediction = T.clip(prediction, eps, 1-eps)
    loss = -(target * T.log(prediction))
    return loss

def weighted_cat_ce(target, prediction, eps=1e-8):
    '''
    This loss weights each class by some factor.
    '''
    prediction = T.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
    target = T.reshape(target, (target.shape[1], target.shape[2]))
    prediction = T.clip(prediction, eps, 1 - eps)
    loss = -((np.array([0.7, 0.3], dtype='float32') / (T.mean(target, axis=0))) * target * T.log(prediction))
    return loss

def cat_ce_parts(target, prediction, eps=1e-8):
    '''
    This loss weights each class by some factor.
    '''
    aleph = 0.4

    prediction = T.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
    target = T.reshape(target, (target.shape[1], target.shape[2]))
    prediction = T.clip(prediction, eps, 1 - eps)

    b_inds = (target[:,1] > 0).nonzero()
    t_bones = target[b_inds]
    p_bones = prediction[b_inds]
    bones_loss = T.mean(-(t_bones * T.log(p_bones)), axis=0, keepdims=True)
    loss = T.mean(-(target * T.log(prediction)), axis=0, keepdims=True)

    return (1 - aleph) * loss + aleph * bones_loss

def dice(target, prediction, eps=1e-8):
    '''
    The dice loss as described in:
    https://arxiv.org/pdf/1606.04797v1.pdf
    (V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation)
    The above paper aims to maximize the dice measure. Since climin only
    minimizes, this function returns 1 - dice instead of dice, with the assumption
    that minimizing the former is equivalent to maximizing the latter.
    '''
    prediction = T.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
    target = T.reshape(target, (target.shape[1], target.shape[2]))
    prediction = T.clip(prediction, eps, 1 - eps)
    loss = 2*T.sum(target*prediction,axis=0,keepdims=True)
    loss /= (T.sum(T.sqr(target),axis=0,keepdims=True) + T.sum(T.sqr(prediction),axis=0,keepdims=True))
    return 1 - loss

def jaccard(target, prediction, eps=1e-8):
    '''
     Jaccard distance, see: https://en.wikipedia.org/wiki/Jaccard_index
    '''
    prediction = T.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
    target = T.reshape(target, (target.shape[1], target.shape[2]))
    prediction = T.clip(prediction, eps, 1 - eps)
    intersection = T.sum(target * prediction, axis=0, keepdims=True)
    loss = intersection / (T.sum(target + prediction, axis=0, keepdims=True) - intersection)
    return 1 - loss

def tanimoto(target, prediction, eps=1e-8):
    '''
    Tanimoto distance, see: https://en.wikipedia.org/wiki/Jaccard_index#Other_definitions_of_Tanimoto_distance
    '''
    prediction = T.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
    target = T.reshape(target, (target.shape[1], target.shape[2]))
    prediction = T.clip(prediction, eps, 1 - eps)

    intersection = T.sum(target * prediction, axis=0, keepdims=True)
    prediction_sq = T.sum(T.sqr(prediction), axis=0, keepdims=True)
    target_sq = T.sum(T.sqr(target), axis=0, keepdims=True)

    loss = intersection / (target_sq + prediction_sq - intersection)
    return 1 - loss

def tanimoto_wmap(target_in, prediction, eps=1e-8):
    '''
    Tanimoto distance, see: https://en.wikipedia.org/wiki/Jaccard_index#Other_definitions_of_Tanimoto_distance
    '''
    target_in = T.reshape(target_in, (target_in.shape[1], target_in.shape[2]))
    target = target_in[:, :2]
    wmap = T.repeat(target_in[:, 2].dimshuffle(('x', 0)), 2, axis=0).dimshuffle((1, 0))
    prediction = T.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
    prediction = T.clip(prediction, eps, 1 - eps)

    target_w = T.sum(T.sqr(target * wmap), axis=0, keepdims=True)
    pred_w = T.sum(T.sqr(prediction * wmap), axis=0, keepdims=True)
    intersection_w = T.sum(target_w * pred_w, axis=0, keepdims=True)

    intersection = T.sum(target * prediction, axis=0, keepdims=True)
    prediction_sq = T.sum(T.sqr(prediction), axis=0, keepdims=True)
    target_sq = T.sum(T.sqr(target), axis=0, keepdims=True)

    loss = (target_w + pred_w - 2 * intersection_w) / (target_sq + prediction_sq - intersection)
    return loss

def kl_divergence(target, prediction, eps=1e-6):
    '''Kullback-Leibler divergence'''
    prediction = T.reshape(prediction, (prediction.shape[1], prediction.shape[2]))
    target = T.reshape(target, (target.shape[1], target.shape[2]))
    prediction = T.clip(prediction, eps, 1 - eps)
    target = T.clip(target, eps, 1 - eps)

    kl = T.sum(target * T.log(target / prediction), axis=0, keepdims=True)
    return kl

class BatchNormFuns(object):
    '''
    Convenience class to compute network forward passes during validation and inference.
    '''
    def __init__(self, model, phase, fun):
        '''
        :param model: network model
        :param phase: 'valid' or 'infer', model will be reset
                       to 'train' after computing forward pass
                       in given phase.
        :param fun: the concrete function of the model to call
                    i.e. model.score or model.predict
        '''
        self.model = model
        self.phase = phase
        self.fun = fun

    def __call__(self, *data):
        return batchnorm_apply_fun(self.model, self.phase, self.fun, data)

def batchnorm_apply_fun(model, phase, fun, data):
    model.phase_select(phase_id=phase)
    res = fun(*data)
    model.reset_phase()
    return res

class PocketTrainer(object):
    def __init__(self, model, data, stop,
                 pause, score_fun, report_fun,
                 evaluate=True, test=True, batchnorm=False,
                 model_code=None, n_report=None):
        self.model = model
        self.data = data
        self.stop = stop
        self.pause = pause
        self.score_fun = score_fun
        self.report_fun = report_fun
        self.best_pars = None
        self.best_loss = float('inf')
        self.runtime = 0
        self.evaluate = evaluate
        self.test = test
        self.losses = []
        self.test_performance = []
        self.model_code = model_code
        self.n_epochs_done = 0
        self.n_iters_done = 0
        self.n_report = n_report

        # if batchnorm:
        #     self.m_score_train = BatchNormFuns(
        #         model=self.model, phase='valid',
        #         fun=self.model.score
        #     )
        #     if bn_mode == 'native':
        #         print 'using batch norm with running metrics for validation'
        #         self.m_score_valid = BatchNormFuns(
        #             model=self.model, phase='valid',
        #             fun=self.model.score
        #         )
        #     elif bn_mode == 'batch':
        #         print 'using batch norm without running metrics'
        #         self.m_score_valid = self.m_score_train
        #     else:
        #         raise ValueError('BN modes are: native, batch')
        # else:
        #     self.m_score_train = self.m_score_valid = self.model.score

        self.using_bn = batchnorm

    def demo(self, predict, image, gt, size_reduction, im_name='test.png'):
        output_h = self.model.image_height - size_reduction
        output_w = self.model.image_width - size_reduction
        output_d = self.model.image_depth - size_reduction
        n_chans = self.model.n_channels
        n_classes = self.model.n_output

        segmentation = predict(image)
        segmentation = segmentation.as_numpy_array() if isinstance(segmentation, gnumpy.garray) else segmentation
        segmentation = np.reshape(
            segmentation,
            (output_h, output_w, output_d, n_classes)
        )

        gt = np.reshape(
            gt, (output_h, output_w, output_d, n_classes)
        )

        dice_list = dice_demo(segmentation, gt)
        segmentation = segmentation.argmax(axis=3)
        gt = gt.argmax(axis=3)

        image = np.reshape(np.transpose(image, (0,2,3,4,1)), (n_chans, output_h, output_w, output_d))
        im_slice = image[0,:,:,image.shape[-1]/2]

        seg_slice = segmentation[:,:,segmentation.shape[-1]/2]
        gt_slice = gt[:,:,gt.shape[-1]/2]

        if n_classes == 2:
            vis_result(im_slice, seg_slice, gt_slice, file_name=im_name)
        elif n_classes == 5:
            vis_col_im(im=im_slice, seg=seg_slice, gt=gt_slice, file_name=im_name)
        else:
            raise NotImplementedError('Can only handle 2 or 5 classes')

        return dice_list

    def fit(self):
        try:
            for i in self.iter_fit(*self.data['train']):
                self.report_fun(i)
        except exceptions.IOError, e:
            pass
        except KeyboardInterrupt:
            self.quit_training()
            import sys
            sys.exit(0)


    def iter_fit(self, *fit_data):
        start = time.time()

        for info in self.model.iter_fit(*fit_data):
            if self.pause(info):
                # Take care of batch norm
                # Things done here shouldn't affect running metrics since no learning is supposed to happen.
                if self.using_bn:
                    self.model.phase_select(phase_id='valid')

                if 'loss' not in info:
                    info['loss'] = ma.scalar(
                        self.score_fun(self.model.score, *self.data['train'])
                    )

                if self.evaluate:
                    info['val_loss'] = ma.scalar(
                        self.score_fun(self.model.score, *self.data['val'])
                    )

                    if info['val_loss'] < self.best_loss:
                        self.best_loss = info['val_loss']
                        self.best_pars = self.model.parameters.data.copy()

                    self.losses.append((info['loss'], info['val_loss']))
                else:
                    self.losses.append(info['loss'])

                if self.test:
                    info['test_avg'] = ma.scalar(
                        self.score_fun(self.model.score, *self.data['test'])
                    )
                    self.test_performance.append(info['test_avg'])

                self.runtime = time.time() - start
                info.update({
                    'best_loss': self.best_loss,
                    'best_pars': self.best_pars,
                    'runtime': self.runtime
                })
                self.n_epochs_done = info['n_iter'] / self.n_report
                self.n_iters_done = info['n_iter']

                # Return to training mode, keep learning running metrics.
                if self.using_bn:
                    self.model.phase_select(phase_id='train')

                yield info

                if self.stop(info):
                    break

    def quit_training(self):
        if self.best_pars is None:
            print 'canceled before the end of the first epoch, nothing to do.'
            return

        model_code = self.model_code
        param_loc = os.path.join('models', 'checkpoints', model_code + '.hdf5')
        GLOB_CKPT_DIR = os.path.join('models', 'checkpoints')
        if not os.path.exists(GLOB_CKPT_DIR):
            os.makedirs(GLOB_CKPT_DIR)
        print 'setting checkpoint at: ', param_loc
        param_file = h5py.File(param_loc, 'w')
        best_params = param_file.create_dataset(
            'best_pars', self.model.parameters.data.shape, dtype='float32'
        )
        last_params = param_file.create_dataset(
            'last_pars', best_params.shape, dtype='float32'
        )

        if isinstance(self.best_pars, gnumpy.garray):
            best_params[...] = self.best_pars.as_numpy_array()
            last_params[...] = self.model.parameters.data.as_numpy_array()
        else:
            best_params[...] = self.best_pars[...]
            last_params[...] = self.model.parameters.data[...]
        param_file.close()

        if self.using_bn:
            bn_pars = self.model.get_batchnorm_params()
            bn_pars_path = os.path.join('models', 'checkpoints', model_code + '_bn_pars.pkl')
            with open(bn_pars_path, 'w') as f:
                pickle.dump(bn_pars, f)

        mini_log_code = os.path.join('models', 'checkpoints', model_code + '_log.json')

        if os.path.exists(mini_log_code):
            print 'previous log found'
            with open(mini_log_code, 'r') as f:
                prev_log = json.load(f)
            print 'updating current log...'

            self.losses = prev_log['losses'] + self.losses
            self.test_performance = prev_log['test_performance'] + self.test_performance
            self.n_epochs_done += prev_log['n_epochs']
            self.n_iters_done += prev_log['n_iters']
        mini_log = {
            'losses': self.losses,
            'test_performance': self.test_performance,
            'best_loss': self.best_loss,
            'n_epochs': self.n_epochs_done,
            'n_iters': self.n_iters_done
        }
        print 'writing new log at: ', mini_log_code
        with open(mini_log_code, 'w') as f:
            json.dump(mini_log, f)

        print 'all done.'
        return

class MinibatchTest(object):
    def __init__(self, max_samples, sample_dims):
        self.max_samples = max_samples
        self.sample_dims = sample_dims

    def __call__(self, predict_f, *data):
        batches = iter_minibatches(data, self.max_samples, self.sample_dims, 1)
        seen_samples = 0.
        score = 0.
        for batch in batches:
            x, z = batch
            y = predict_f(x)
            this_samples = int(y.shape[self.sample_dims[0]])
            errs = (y.argmax(axis=1) != z.argmax(axis=1)).sum()
            score += errs
            seen_samples += this_samples

        return ma.scalar(score / seen_samples)

class MinibatchTestFCN(object):
    def __init__(self, max_samples, sample_dims):
        self.max_samples = max_samples
        self.sample_dims = sample_dims

    def __call__(self, predict_f, *data):
        batches = iter_minibatches(data, self.max_samples, self.sample_dims, 1)
        seen_samples = 0.
        score = 0.
        for batch in batches:
            x, z = batch
            y = predict_f(x)
            this_samples = int(y.shape[1])
            errs = (y.argmax(axis=2) != z.argmax(axis=2)).sum()
            score += errs
            seen_samples += this_samples

        return ma.scalar(score / seen_samples)

class MinibatchScoreFCN(object):
    def __init__(self, max_samples, sample_dims):
        self.max_samples = max_samples
        self.sample_dims = sample_dims

    def __call__(self, f_score, *data):
        batches = iter_minibatches(data, self.max_samples, self.sample_dims, 1)
        score = 0.
        seen_samples = 0.
        for batch in batches:
            x = batch[0]
            z = batch[1]
            this_samples = int(x.shape[0])
            score += f_score(x, z) * this_samples
            seen_samples += this_samples
        return ma.scalar(score / seen_samples)