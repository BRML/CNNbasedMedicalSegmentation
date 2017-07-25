import cPickle as pickle

import matplotlib.pyplot as plt
import numpy as np
import h5py

import climin.stops
from climin import mathadapt as ma

from breze.learn.trainer import report

import ash
from ash import PocketTrainer

from conv3d.model import FCN

vis = False
params = False
stacked_filters = False

d_code = '96fcnmini'
param_file = 'params/params' + d_code + '.pkl'
f = h5py.File('data/data'+d_code+'.hdf5', 'r')

train_x = f['train_x']
train_y = f['train_y']

valid_x = f['valid_x']
valid_y = f['valid_y']

test_x = f['test_x']
test_y = f['test_y']

n_classes = 2

if d_code == '48fcnmini' or d_code == '48fcn' and stacked_filters:
    nkerns_d = [16, 16, 32, 32, 64, 64]
    fs_d = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    ps_d = ['no_pool', (2, 2, 2), 'no_pool', (2, 2, 2), 'no_pool', (2, 2, 2)]
    strides_d = (1, 1, 1)

    nkerns_u = [128, 128, 64, 64, 32, 32, 16, n_classes]
    fs_u = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)]
    ps_u = ['no_pool', (2, 2, 2), 'no_pool', (2, 2, 2), 'no_pool', (2, 2, 2), 'no_pool', 'no_pool']

    hidden_transfers_conv = ['rectifier']*6
    hidden_transfers_upconv = ['rectifier']*7 + ['identity']

    bm_down = ['same']*6
    bm_up = ['same']*7 + ['valid']
    loss_id = ash.fcn_cat_ce

    padding = 0
elif d_code.startswith('48fcn'):
    nkerns_d = [2, 16, 32]
    fs_d = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
    ps_d = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
    strides_d = (1, 1, 1)

    nkerns_u = [32, 32, 16, n_classes]
    fs_u = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    ps_u = [(2, 2, 2), (2, 2, 2), (2, 2, 2), 'no_pool']

    hidden_transfers_conv = ['rectifier', 'rectifier', 'rectifier']
    hidden_transfers_upconv = ['rectifier', 'rectifier', 'rectifier', 'identity']

    bm_down = ['same', 'same', 'same']
    bm_up = ['same', 'same', 'same', 'same']
    loss_id = ash.fcn_cat_ce

    padding = 0
elif d_code.startswith('9680fcn') and stacked_filters:
    nkerns_d = [16, 16, 32, 32, 64, 64]
    fs_d = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    ps_d = ['no_pool', (2, 2, 2), 'no_pool', (2, 2, 2), 'no_pool', (2, 2, 2)]
    strides_d = (1, 1, 1)

    nkerns_u = [128, 128, 64, 64, 32, 32, 16, n_classes]
    fs_u = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)]
    ps_u = ['no_pool', (2, 2, 2), 'no_pool', (2, 2, 2), 'no_pool', (2, 2, 2), 'no_pool', 'no_pool']

    hidden_transfers_conv = ['rectifier']*6
    hidden_transfers_upconv = ['rectifier']*7 + ['identity']

    bm_down = ['same']*6
    bm_up = ['valid']*2 + ['same']*5 + ['valid']
    loss_id = ash.fcn_cat_ce

    padding = 16
elif d_code.startswith('9680fcn'):
    nkerns_d = [8, 16, 32]
    fs_d = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
    ps_d = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
    strides_d = (1, 1, 1)

    nkerns_u = [32, 32, 16, n_classes]
    fs_u = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    ps_u = [(2, 2, 2), (2, 2, 2), (2, 2, 2), 'no_pool']

    hidden_transfers_conv = ['rectifier', 'rectifier', 'rectifier']
    hidden_transfers_upconv = ['rectifier', 'rectifier', 'rectifier', 'identity']

    bm_down = ['same', 'same', 'same']
    bm_up = ['valid', 'same', 'same', 'same']
    loss_id = ash.fcn_cat_ce

    padding = 16
elif d_code.startswith('96fcn'):
    nkerns_d = [8, 16, 32]
    fs_d = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
    ps_d = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
    strides_d = (1, 1, 1)

    nkerns_u = [32, 32, 16, n_classes]
    fs_u = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    ps_u = [(2, 2, 2), (2, 2, 2), (2, 2, 2), 'no_pool']

    hidden_transfers_conv = ['rectifier', 'rectifier', 'rectifier']
    hidden_transfers_upconv = ['rectifier', 'rectifier', 'rectifier', 'identity']

    bm_down = ['same', 'same', 'same']
    bm_up = ['same', 'same', 'same', 'same']
    loss_id = ash.dice

    padding = 0
else:
    raise Exception('No such dataset.')

train_size, inpt_d, n_channels, inpt_h, inpt_w = train_x.shape

batch_size = 1
max_passes = 50

set_x = train_x
set_y = train_y

output_h = inpt_h-padding
output_w = inpt_w-padding
output_d = inpt_d-padding

if vis:
    for i in range(set_x.shape[0]):
        plt.imshow(set_x[i,inpt_d/2,0,:,:], cmap='Greys_r')
        plt.show()
        ty = np.reshape(set_y[i], (output_h,output_w,output_d,n_classes))
        ty = ty.argmax(axis=3)
        plt.imshow(ty[:,:,output_d/2], cmap='Greys_r')
        plt.show()

inpt_dims = (inpt_h, inpt_w, inpt_d)

n_report = train_size / batch_size
max_iter = n_report * max_passes

print 'Train x shape: ', train_x.shape
print 'Train y shape: ', train_y.shape
print 'Valid x shape: ', valid_x.shape
print 'Valid y shape: ', valid_y.shape
print 'Test x shape: ', test_x.shape
print 'Test y shape: ', test_y.shape

print '\nmax iter: ', max_iter
print 'report frequency: every %i iterations\n' % n_report

stop = climin.stops.AfterNIterations(max_iter)
pause = climin.stops.ModuloNIterations(n_report)

optimizer = 'adam'

pkchu = FCN(
    image_height=inpt_dims[0], image_width=inpt_dims[1],
    image_depth=inpt_dims[2], n_channel=n_channels,
    n_output=n_classes, n_hiddens_conv=nkerns_d,
    down_filter_shapes=fs_d, hidden_transfers_conv=hidden_transfers_conv,
    down_pools=ps_d, n_hiddens_upconv=nkerns_u,
    up_filter_shapes=fs_u, hidden_transfers_upconv=hidden_transfers_upconv,
    up_pools=ps_u, out_transfer='softmax', loss=loss_id,
    optimizer=optimizer, batch_size=batch_size,
    bm_up=bm_up, bm_down=bm_down,
    max_iter=max_iter, implementation='dnn_conv3d', strides_d=strides_d
)

print '\nARCHITECTURE: '
print '\tFilters: ', fs_d
print '\tFeature maps: ', nkerns_d
print '\tPools: ', ps_d
print '\tUp-filters: ', fs_u
print '\tFeature maps: ', nkerns_u
print '\tUppools: ', ps_u

if not params:
    rng = np.random.RandomState(123)
    pkchu.parameters.data[...] = rng.normal(0, 0.01, pkchu.parameters.data.shape)
else:
    with open(param_file, 'r') as f:
    #with open('params/params9680fcnmini.pkl', 'r') as f:
        pkchu.parameters.data[...] = pickle.load(f)

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

initial_err = ma.scalar(score_fun(pkchu.score, *data['train']))

print 'Initial train loss: %.4f' % initial_err

coach = PocketTrainer(
    model=pkchu, data=data, stop=stop,
    pause=pause, score_fun=score_fun,
    report_fun=report_fun, test_fun=test_fun,
    evaluate=True, test=True
)

coach.fit()

pkchu.parameters.data[...] = coach.best_pars

plt.plot(coach.losses)
plt.show()

for i in range(test_x.shape[0]):
    plt.imshow(test_x[i, inpt_d/2, 0, :, :], cmap='Greys_r')
    plt.show()
    y = pkchu.predict(test_x[i:i+1])
    y = np.reshape(y[0].as_numpy_array(), (output_h, output_w, output_d, n_classes))
    y = y.argmax(axis=3)
    plt.imshow(y[:, :, output_d/2], cmap='Greys_r')
    plt.show()

with open(param_file,'w') as f:
    pickle.dump(coach.best_pars, f)