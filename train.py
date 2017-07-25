"""
This is the script used for training one of the neural networks
defined in model_defs.py
The usage is:

python train.py [model_id] [dataset] [save_folder] [n_epoch] -ch (True/False)

[model_id] is the id of the model used in model_defs.py
[dataset] is the name of a dataset from data/datasets
[save_folder] our results will be saved to models/[save_folder]
[n_epoch] We will iterate over the entire data set [n_epoch] many times

-ch: Flag indicating whether we want to start training from an earlier checkpoint
     WARNING: checkpoints are specific to the model_id and not to the experiment.
              If you have two different experiments using the same model_id running
              in parallel, their checkpoints will be in conflict.

In our paper, we trained our network using:

python train.py fcn_rffc4 brats_fold0 brats_fold0 600 -ch False
"""

#import break_handling
import cPickle as pickle
import json
import os
import datetime
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import gnumpy
import h5py

import climin.stops
from breze.learn.trainer import report

import ash
from ash import PocketTrainer
from model_defs import get_model

from conv3d.model import SequentialModel

def make_parser():
    parser = argparse.ArgumentParser(description='Train model on data.')
    parser.add_argument('model_code', metavar='model', type=str, help='model to use')
    parser.add_argument('data_code', metavar='data', type=str, help='data to train on')
    parser.add_argument('train_code', metavar='tdir', type=str, help='directory path to store results in')
    parser.add_argument('n_epochs', metavar='ne', type=int, help='num of passes through the training set')
    parser.add_argument('-ch', '--checkpoint', help='set to load from checkpoint if available')

    return parser

def retrieve_data(data_code):
    data_loc = os.path.join('data', 'datasets', data_code + '.hdf5')
    data = h5py.File(data_loc, 'r')

    train_x = data['train_x']
    train_y = data['train_y']
    valid_x = data['valid_x']
    valid_y = data['valid_y']
    test_x = data['test_x']
    test_y = data['test_y']

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

def load_checkpoint(model_code, param_shape):
    param_loc = os.path.join('models', 'checkpoints', model_code + '.hdf5')
    log_code = os.path.join('models', 'checkpoints', model_code + '_log.json')
    bn_pars_path = os.path.join('models', 'checkpoints', model_code + '_bn_pars.pkl')
    bn_pars = None
    if not os.path.exists(param_loc):
        print 'No checkpoint available, using random initialization instead.'
        return np.random.normal(0, 0.01, param_shape), None
    else:
        with open(log_code, 'r') as f:
            log = json.load(f)
            n_epochs_done = log['n_epochs']
            n_iters_done = log['n_iters']
            b_loss = log['best_loss']
        if os.path.exists(bn_pars_path):
            with open(bn_pars_path, 'r') as f:
                bn_pars = pickle.load(f)
                print 'bn parameters found'
        param_file = h5py.File(param_loc, 'r')
        params_np = np.zeros(param_shape)
        b_params_np = np.zeros(param_shape)
        params_np[...] = param_file['last_pars']
        b_params_np[...] = param_file['best_pars']
        param_file.close()
        t_dic = {
            'best_pars': b_params_np,
            'best_loss': b_loss,
            'n_epochs': n_epochs_done,
            'n_iters': n_iters_done,
            'bn_pars': bn_pars
        }
        return params_np, t_dic

def build_model(model_code, checkpoint, info):
    model_def = get_model(model_code=model_code)

    layer_vars = model_def.layer_vars
    batchnorm = model_def.batchnorm
    loss_id = model_def.loss_id
    loss_layer_def = model_def.loss_layer_def
    out_transfer = model_def.out_transfer

    if model_def.regularize:
        print 'using regularization: l1: %s, l2: %s' % (model_def.l1, model_def.l2)

    model = SequentialModel(
        image_height=info['height'], image_width=info['width'],
        image_depth=info['depth'], n_channels=info['n_inpt'],
        n_output=info['n_classes'], layer_vars=layer_vars,
        out_transfer=out_transfer, loss_id=loss_id,
        loss_layer_def=loss_layer_def, optimizer=info['optimizer'],
        batch_size=info['batch_size'], max_iter=info['max_iter'],
        using_bn=batchnorm, regularize=model_def.regularize,
        l1=model_def.l1, l2=model_def.l2,
        perform_transform=model_def.perform_transform
    )

    if checkpoint:
        model.parameters.data[...], t_dic = load_checkpoint(model_code, model.parameters.data.shape)
        if t_dic is not None:
            model.max_iter -= t_dic['n_iters']
            if t_dic['bn_pars'] is not None:
                model.set_batchnorm_params(t_dic['bn_pars'])
                print 'bn parameters loaded'
    else:
        t_dic = None
        rng = np.random.RandomState(123)
        model.parameters.data[...] = rng.normal(0, 0.01, model.parameters.data.shape)

    return model, model_def, t_dic

def setup_training(model_code, data_code, checkpoint, max_passes):
    train, valid, test = retrieve_data(data_code=data_code)

    train_size, inpt_d, n_channels, inpt_h, inpt_w = train[0].shape
    n_classes = train[1].shape[-1]
    valid_size = valid[0].shape[0]
    test_size = test[0].shape[0]

    print 'input data dimensions: h: %i w: %i d: %i' % (inpt_h, inpt_w, inpt_d)
    print 'set stats: train: %i, valid: %i, test: %i' % (train_size, valid_size, test_size)

    optimizer = 'adam'
    batch_size = 1

    n_report = train_size / batch_size
    max_iter = n_report * max_passes

    info = {
        'height':inpt_h, 'width': inpt_w, 'depth': inpt_d,
        'n_classes': n_classes, 'n_inpt': n_channels, 'optimizer': optimizer,
        'batch_size': batch_size, 'max_iter': max_iter, 'n_report': n_report
    }

    model, model_def, t_dic = build_model(model_code, checkpoint, info)

    stop = climin.stops.AfterNIterations(max_iter=model.max_iter)
    pause = climin.stops.ModuloNIterations(n_report)

    data = {
        'train': train, 'val': valid, 'test': test
    }

    report_fun = report.OneLinePrinter(
        ['n_iter', 'runtime', 'loss', 'val_loss'],
        spaces=['4', '7.4f', '5.4f', '7.4f']
    )
    score_fun = ash.MinibatchScoreFCN(max_samples=batch_size, sample_dims=[0, 0])

    coach = PocketTrainer(
        model=model, data=data, stop=stop,
        pause=pause, score_fun=score_fun,
        report_fun=report_fun, evaluate=True,
        test=False, batchnorm=model_def.batchnorm,
        model_code=model_code, n_report=n_report
    )

    if t_dic is not None:
        coach.best_pars = t_dic['best_pars'].copy()
        coach.best_loss = t_dic['best_loss']

    return coach, model_def

def secure_data(coach, params_shape, model_def, train_dir):
    param_loc = os.path.join(train_dir, 'params.hdf5')
    param_file = h5py.File(param_loc, 'w')
    model_params = param_file.create_dataset(
        'best_pars', params_shape, dtype='float32'
    )

    if isinstance(coach.best_pars, gnumpy.garray):
        model_params[...] = coach.best_pars.as_numpy_array()
    else:
        model_params[...] = coach.best_pars

    if coach.using_bn:
        bn_pars = coach.model.get_batchnorm_params()
        bn_pars_loc = os.path.join(train_dir, 'bn_pars.pkl')
        with open(bn_pars_loc, 'w') as f:
            pickle.dump(bn_pars, f)

    now = str(datetime.datetime.now())
    date, time = now.split(' ')
    time = time.replace(':', '_')
    time = time.replace('.', '_')
    log_code = 'log' + date + '@' + time + '.json'
    log_loc = os.path.join(train_dir, log_code)

    log = {
        'params': param_loc,
        'layers': model_def.layer_vars,
        'loss_id': model_def.loss_name,
        'losses': coach.losses,
        'test_performance': coach.test_performance,
        'regularize': model_def.regularize,
        'l1': model_def.l1,
        'l2': model_def.l2,
        'perform_transform': model_def.perform_transform.__name__ if model_def.perform_transform is not None else None
    }

    with open(log_loc, 'w') as f:
        json.dump(log, f)

    t_loss, v_loss = plt.plot(coach.losses)
    plt.legend([t_loss, v_loss], ['train loss', 'val loss'])
    save_file = os.path.join(train_dir, 'figure.png')
    plt.savefig(save_file)

def save_demo(coach, train_dir, size_reduction):
    if coach.using_bn:
        predict = ash.BatchNormFuns(
            model=coach.model,
            fun=coach.model.predict,
            phase='infer'
        )
    else:
        predict = coach.model.predict

    test_x, test_y = coach.data['test']
    dice_values = []
    for i in range(test_x.shape[0]):
        im_name = os.path.join(train_dir, 'im' + str(i) + '.png')
        this_dice_value = coach.demo(
            predict=predict, image=test_x[i:i + 1],
            gt=test_y[i], size_reduction=size_reduction,
            im_name=im_name
        )
        dice_values.append(this_dice_value)

    mean_dice = 0
    for d in dice_values:
        mean_dice += d[0]
    mean_dice = mean_dice * 1./len(dice_values)
    dice_log = {'mean_dice': mean_dice, 'dice_values': dice_values}
    dice_log_path = os.path.join(train_dir, 'dice.json')
    with open(dice_log_path, 'w') as f:
        json.dump(dice_log, f)

def start_training(model_code, data_code, checkpoint, max_passes, train_dir=None):
    if train_dir is None:
        train_dir = os.path.join('models', data_code)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    print 'Building model, coach...'
    coach, model_def = setup_training(model_code, data_code, checkpoint, max_passes)

    #break_handling.make_checkpoint = coach.quit_training
    print 'Starting training...'
    coach.fit()

    print 'Securing results...'
    secure_data(
        coach=coach, params_shape=coach.model.parameters.data.shape,
        model_def=model_def, train_dir=train_dir
    )

    print 'Saving demo images...'
    save_demo(coach, train_dir, size_reduction=model_def.size_reduction)

    print 'done.'

if __name__ == '__main__':
    parser = make_parser()

    if len(sys.argv[1:]) > 0:
        args = parser.parse_args()
    else:
        args = parser.parse_args(['fcn_rffc4', 'dummy5', 'dummy5', '2', '-ch', 'True']) # model data train checkpoint

    model_code = args.model_code
    data_code = args.data_code
    checkpoint = args.checkpoint
    t_code = args.train_code
    n_epochs = args.n_epochs

    train_dir = os.path.join('models', t_code)
    start_training(model_code, data_code, checkpoint, max_passes=n_epochs, train_dir=train_dir)