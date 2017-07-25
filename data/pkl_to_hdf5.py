import cPickle as pickle
import gzip
import numpy as np
import h5py

import matplotlib.pyplot as plt

from breze.learn.data import one_hot

#
# The following needs to be adapted
# to the particular set of patch pickles
# being read:
# the name of the .hdf5 file, b_size, train_size,
# test_size and valid_size, dims, gt_dims, p_code(located right 
# before the for-loop.) 

h_file = h5py.File('data96fcnmini.hdf5', 'w')

b_size = 5
train_size = 20*b_size
valid_size = test_size = 5*b_size
dims = (96, 96, 96)
gt_dims = (dims[0],dims[1],dims[2])
dimprod = np.prod(np.array(dims))
gt_dimprod = np.prod(np.array(gt_dims))

print 'Train size: ', train_size
print 'Valid size: ', valid_size
print 'Test size: ', test_size

full_x = np.zeros((train_size+valid_size+test_size, dims[2], 1, dims[0], dims[1]), dtype='float32')
full_y = np.zeros((train_size+valid_size+test_size, gt_dimprod, 2), dtype='float32')

###### DELETE THIS LATER ######
#train_size = 2*b_size
#valid_size = test_size = b_size
###### DELETE THIS LATER ######

train_x = h_file.create_dataset(
    'train_x', (train_size, dims[2], 1, dims[0], dims[1]), dtype='float32'
)
train_y = h_file.create_dataset(
    'train_y', (train_size,gt_dimprod,2), dtype='float32'
)

valid_x = h_file.create_dataset(
    'valid_x', (valid_size, dims[2], 1, dims[0], dims[1]), dtype='float32'
)
valid_y = h_file.create_dataset(
    'valid_y', (valid_size,gt_dimprod,2), dtype='float32'
)

test_x = h_file.create_dataset(
    'test_x', (test_size, dims[2], 1, dims[0], dims[1]), dtype='float32'
)
test_y = h_file.create_dataset(
    'test_y', (test_size,gt_dimprod,2), dtype='float32'
)

count = 1
index = train_i = valid_i = test_i = 0
p_code = 'fcn96'

while count <= 30:
    print 'Reading image ', count
    with gzip.open('../../patches'+p_code+'/im'+str(count)+'.pkl.gz', 'rb') as f:
        patches, labels = pickle.load(f)
    print labels.shape
    for lbl in labels:
        print 'Mean value: ', lbl.mean()
    labels = np.reshape(labels, (-1,))
    labels = np.asarray(labels, dtype='int16')
    full_x[index:index+b_size,:,0,:,:] = np.transpose(patches, (0, 3, 1, 2))
    l = np.reshape(one_hot(labels, 2), (b_size, gt_dimprod, -1))
    full_y[index:index+b_size,:,:] = np.asarray(l, dtype='float32')
    plt.imshow(full_x[index,dims[2]/2,0,:,:], cmap='Greys_r')
    plt.show()
    gt = np.reshape(labels[:gt_dimprod], gt_dims)
    plt.imshow(gt[:,:,gt_dims[2]/2], cmap='Greys_r')
    plt.show()
    index += b_size
    count += 1

#rand_indices = np.random.permutation(train_size+valid_size+test_size)
rand_indices = np.arange(train_size+valid_size+test_size)

train_x[:,:,:,:,:] = full_x[rand_indices[:train_size],:,:,:,:]
train_y[:,:,:] = full_y[rand_indices[:train_size],:,:]

valid_x[:,:,:,:,:] = full_x[rand_indices[train_size:train_size+valid_size],:,:,:,:]
valid_y[:,:,:] = full_y[rand_indices[train_size:train_size+valid_size],:,:]

test_x[:,:,:,:,:] = full_x[rand_indices[train_size+valid_size:],:,:,:,:]
test_y[:,:,:] = full_y[rand_indices[train_size+valid_size:],:,:]

#while count <= 30:
#    print 'Reading image ', count
#    with gzip.open('../../patches'+p_code+'/im'+str(count)+'.pkl.gz', 'rb') as f:
#        patches, labels = pickle.load(f)
#    rand_indices = np.random.permutation(b_size)
#    patches = patches[rand_indices,:,:,:]
#    labels = labels[rand_indices,:]
#    labels = np.reshape(labels, (-1,)) 
#    labels = np.asarray(labels, dtype='int16')
#    
#    if count <= 20:
#        train_x[train_i:train_i+b_size,:,0,:,:] = np.transpose(patches, (0, 3, 1, 2))
#	    l = np.reshape(one_hot(labels, 2),(b_size,dimprod,-1))
#        train_y[train_i:train_i+b_size,:,:] = np.asarray(l, dtype='float32')
#        train_i += b_size
#    elif count <= 25:
#        valid_x[valid_i:valid_i+b_size,:,0,:,:] = np.transpose(patches, (0, 3, 1, 2))
#	    l = np.reshape(one_hot(labels, 2),(b_size,dimprod,-1))
#        valid_y[valid_i:valid_i+b_size,:,:] = np.asarray(l, dtype='float32')
#        valid_i += b_size
#    else:
#        test_x[test_i:test_i+b_size,:,0,:,:] = np.transpose(patches, (0, 3, 1, 2))
#	    l = np.reshape(one_hot(labels, 2),(b_size,dimprod,-1))
#        test_y[test_i:test_i+b_size,:,:] = np.asarray(l, dtype='float32')
#        test_i += b_size
#    print '\tdone.'
#    count += 1
