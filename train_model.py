import cPickle as pickle
import json
import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import h5py

import climin.stops
from climin import mathadapt as ma

from breze.learn.trainer import report

import ash
from ash import PocketTrainer
from model_defs import get_model

from conv3d.model import SequentialModel

vis = False
retrain = False

d_code = 'handsize2_v2'
model_code = 'fcn96_rescaled'
train_dir = os.path.join('models', d_code)
assert os.path.exists(train_dir)

now = str(datetime.datetime.now())
date, time = now.split(' ')
time = time.replace(':', '_')
time = time.replace('.', '_')
log_code = 'log' + date + '@' + time + '.json'

param_file = os.path.join(train_dir, 'params.hdf5')
f = h5py.File('data/datasets/'+d_code+'.hdf5', 'r')

train_x = f['train_x']
train_y = f['train_y']

valid_x = f['valid_x']
valid_y = f['valid_y']

test_x = f['test_x']
test_y = f['test_y']

n_classes = 2

model_def = get_model(model_code)

alpha = model_def.alpha
layer_vars = model_def.layer_vars
batchnorm = model_def.batchnorm
loss_id = model_def.loss_id
out_transfer = model_def.out_transfer
size_reduction = model_def.size_reduction

train_size, inpt_d, n_channels, inpt_h, inpt_w = train_x.shape

set_x = train_x
set_y = train_y

output_h = inpt_h-size_reduction
output_w = inpt_w-size_reduction
output_d = inpt_d-size_reduction

if vis:
    for i in range(set_x.shape[0]):
        plt.imshow(set_x[i,inpt_d/2,0,:,:], cmap='Greys_r')
        plt.show()
        ty = np.reshape(set_y[i], (output_h,output_w,output_d,n_classes))
        ty = ty.argmax(axis=3)
        plt.imshow(ty[:,:,output_d/2], cmap='Greys_r')
        plt.show()

batch_size = 1
max_passes = 20
inpt_dims = (inpt_h, inpt_w, inpt_d)

n_report = train_size / batch_size
max_iter = n_report * max_passes

#stop = climin.stops.Patience(
#    func_or_key='val_loss', initial=max_iter,
#    grow_factor=2., grow_offset=0,
#    threshold=1e-4
#)
stop = climin.stops.AfterNIterations(max_iter=max_iter)
pause = climin.stops.ModuloNIterations(n_report)

print 'Input data dimensions: h: %i w: %i d: %i ' % (inpt_h, inpt_w, inpt_d)
print 'Set stats: train: %i, valid: %i, test: %i' % (train_x.shape[0], valid_x.shape[0], test_x.shape[0])

print 'max iter: ', max_iter
print 'report frequency: every %i iterations' % n_report

optimizer = 'adam'

print '\nbuilding model...'
pkchu = SequentialModel(
    image_height=inpt_dims[0], image_width=inpt_dims[1],
    image_depth=inpt_dims[2], n_channels=n_channels,
    n_output=n_classes, layer_vars=layer_vars,
    out_transfer=out_transfer, loss_id=loss_id,
    optimizer=optimizer, batch_size=batch_size,
    max_iter=max_iter, using_bn=batchnorm
)

rng = np.random.RandomState(123)
pkchu.parameters.data[...] = rng.normal(0, 0.01, pkchu.parameters.data.shape)

if retrain:
    print 'retrieving old params...'
    f_params = h5py.File(param_file, 'r')
    pkchu.parameters.data[...] = f_params['best_pars']

    if batchnorm:
        bn_par_file = os.path.join(train_dir, 'bn_pars.pkl')
        with open(bn_par_file, 'r') as f:
            bn_pars = pickle.load(f)
            pkchu.set_batchnorm_params(bn_pars)

    param_file = os.path.join(train_dir, 'newparams.hdf5')

report_fun = report.OneLinePrinter(
    ['n_iter', 'runtime', 'loss', 'val_loss', 'test_avg'],
    spaces=['4', '7.4f', '5.4f', '7.4f', '7.4f']
)

score_fun = ash.MinibatchScoreFCN(max_samples=batch_size, sample_dims=[0, 0])
data = {
    'train':(train_x, train_y),
    'val':(valid_x, valid_y),
    'test':(test_x, test_y)
}

test_fun = ash.MinibatchTestFCN(max_samples=batch_size, sample_dims=[0, 0])

#initial_err = ma.scalar(score_fun(pkchu.score, *data['train']))
#print 'Initial train loss: %.4f' % initial_err

coach = PocketTrainer(
    model=pkchu, data=data, stop=stop,
    pause=pause, score_fun=score_fun,
    report_fun=report_fun, test_fun=test_fun,
    evaluate=True, test=True, batchnorm=batchnorm
)

print 'training...'
coach.fit()
print 'training complete.'

pkchu.parameters.data[...] = coach.best_pars

f_params = h5py.File(param_file, 'w')
model_params = f_params.create_dataset(
    'best_pars', pkchu.parameters.data.shape, dtype='float32'
)

print 'securing params...'
model_params[...] = coach.best_pars.as_numpy_array()

if batchnorm:
    print 'securing batch-norm params...'
    bn_pars = pkchu.get_batchnorm_params()
    with open(os.path.join(train_dir, 'bn_pars.pkl'), 'w') as f:
        pickle.dump(bn_pars, f)

log = {
    'data': d_code,
    'params': param_file,
    'layers': layer_vars,
    'loss_id': loss_id.__name__,
    'losses': coach.losses,
    'test_performance': coach.test_performance
}

print 'printing log and visualizing results...'
with open(os.path.join(train_dir, log_code), 'w') as f:
    json.dump(log, f)

t_loss, v_loss = plt.plot(coach.losses)
plt.legend([t_loss, v_loss], ['train loss', 'val loss'])
save_file = os.path.join(train_dir, 'figure.png')
plt.savefig(save_file)
#plt.show()

predict = ash.BatchNormFuns(
    model=pkchu,
    fun=pkchu.predict,
    phase='infer'
)

for i in range(test_x.shape[0]):
    coach.demo(predict=predict, image=test_x[i:i+1], gt=test_y[i], size_reduction=size_reduction)
print 'all done, good night.'

