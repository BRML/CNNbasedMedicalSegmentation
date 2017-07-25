import os
import numpy as np
import h5py

from breze.learn.data import one_hot

DATA_HOME = 'datasets'

# Make sure directory 'datasets' exists:
if not os.path.exists(DATA_HOME):
	os.makedirs(DATA_HOME)

ds = (64, 80, 72)
dp = np.prod(np.array(ds))
n_chans = 4
x_size = 2
v_size = t_size = 1

n_classes = 5

x = np.random.randn(x_size, ds[2], 4, ds[0], ds[1])
y = np.random.randint(low=0,high=n_classes,size=(x_size, dp))
y = one_hot(np.reshape(y, (-1,)), n_classes)
y = np.reshape(y, (x_size, dp, n_classes))

vx = np.random.randn(v_size, ds[2], 4, ds[0], ds[1])
vy = np.random.randint(low=0,high=n_classes,size=(1, dp))
vy = one_hot(np.reshape(vy, (-1,)), n_classes)
vy = np.reshape(vy, (v_size, dp, n_classes))

tx = np.random.randn(t_size, ds[2], 4, ds[0], ds[1])
ty = np.random.randint(low=0,high=n_classes,size=(1, dp))
ty = one_hot(np.reshape(ty, (-1,)), n_classes)
ty = np.reshape(ty, (t_size, dp, n_classes))

f = h5py.File(os.path.join(DATA_HOME, 'dummy45.hdf5'), 'w')
train_x = f.create_dataset('train_x', x.shape, dtype='float32')
train_y = f.create_dataset('train_y', y.shape, dtype='float32')
valid_x = f.create_dataset('valid_x', vx.shape, dtype='float32')
valid_y = f.create_dataset('valid_y', vy.shape, dtype='float32')
test_x = f.create_dataset('test_x', tx.shape, dtype='float32')
test_y = f.create_dataset('test_y', ty.shape, dtype='float32')

print 'writing to dummy...'
train_x[...] = x
train_y[...] = y
valid_x[...] = vx
valid_y[...] = vy
test_x[...] = tx
test_y[...] = ty
print 'done.'