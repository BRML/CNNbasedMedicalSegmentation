import cPickle as pickle
import gzip
import numpy as np
import h5py
import json
import os

import matplotlib.pyplot as plt

from breze.learn.data import one_hot

d_code = 'handsize2_v2'
data_dir = 'datasets'
hdf_code = d_code + 'fcn.hdf5'
data = os.path.join(data_dir, hdf_code)
h_file = h5py.File('datasets//handsize2_v2.hdf5', 'w')

b_size = 6
train_n = 20
valid_n = 5
train_size = train_n*b_size
valid_size = test_size = valid_n*b_size

#d = int(d_code)
dims = (64, 64, 192)
dimprod = dims[0] * dims[1] * dims[2]

print 'Train size: ', train_size
print 'Valid size: ', valid_size
print 'Test size: ', test_size

full_x = np.zeros((30, b_size, dims[2], 1, dims[0], dims[1]), dtype='float32')
full_y = np.zeros((30, b_size, dimprod, 2), dtype='float32')

train_x = h_file.create_dataset(
    'train_x', (train_size, dims[2], 1, dims[0], dims[1]), dtype='float32'
)
train_y = h_file.create_dataset(
    'train_y', (train_size,dimprod,2), dtype='float32'
)

valid_x = h_file.create_dataset(
    'valid_x', (valid_size, dims[2], 1, dims[0], dims[1]), dtype='float32'
)
valid_y = h_file.create_dataset(
    'valid_y', (valid_size,dimprod,2), dtype='float32'
)

test_x = h_file.create_dataset(
    'test_x', (test_size, dims[2], 1, dims[0], dims[1]), dtype='float32'
)
test_y = h_file.create_dataset(
    'test_y', (test_size,dimprod,2), dtype='float32'
)

count = 1
index = 0
p_code = 'fcn' + d_code

means = np.zeros((30,b_size))

def shuffle_ims(ims, gts):
    shuffle_indices = np.random.permutation(ims.shape[0])

    ims = ims[shuffle_indices]
    gts = gts[shuffle_indices]
    return ims, gts

def z_mirror_ims(ims, gts):
    rand_indices = np.random.randint(low=0, high=ims.shape[0], size=(ims.shape[0]/2,))
    ims[rand_indices] = ims[rand_indices,:,:,::-1]

    gt_rands = np.reshape(
        gts[rand_indices], (len(rand_indices), ims.shape[1], ims.shape[2], ims.shape[3])
    )
    gt_rands = gt_rands[:,:,:,::-1]
    gts[rand_indices] = np.reshape(gt_rands, (len(rand_indices),-1))

    return ims, gts   

while count <= 30:
    print 'Reading image ', count
    #with gzip.open('../patches/patches'+p_code+'/im'+str(count)+'.pkl.gz', 'rb') as f:
    with gzip.open('../patches/noisy_bi_images/im'+str(count)+'.pkl.gz', 'rb') as f:
        patches, labels = pickle.load(f)
    
    patches, labels = shuffle_ims(patches, labels)
    patches, labels = z_mirror_ims(patches, labels)
    print labels.shape
    for i, lbl in enumerate(labels):
        mean = lbl.mean()
        means[count-1,i] = mean
        print 'Mean value: ', mean
    labels = np.reshape(labels, (-1,))
    labels = np.asarray(labels, dtype='int16')
    full_x[index,:,:,0,:,:] = np.transpose(patches, (0, 3, 1, 2))
    l = np.reshape(one_hot(labels, 2), (b_size, dimprod, -1))
    full_y[index,:,:,:] = np.asarray(l, dtype='float32')
    #plt.imshow(full_x[index,dims[2]/2,0,:,:], cmap='Greys_r')
    #plt.show()
    gt = np.reshape(labels[:dimprod], dims)
    #plt.imshow(gt[:,:,dims[2]/2], cmap='Greys_r')
    #plt.show()
    index += 1
    count += 1

rand_indices = np.random.permutation(30)
#rand_indices = np.arange(30)

train_x[:,:,:,:,:] = np.reshape(full_x[rand_indices[:train_n],:,:,:,:,:], train_x.shape)
train_y[:,:,:] = np.reshape(full_y[rand_indices[:train_n],:,:,:], train_y.shape)
train_mean = means[rand_indices[:train_n],:]
train_mean = train_mean.mean()

valid_x[:,:,:,:,:] = np.reshape(full_x[rand_indices[train_n:train_n+valid_n],:,:,:,:,:], valid_x.shape)
valid_y[:,:,:] = np.reshape(full_y[rand_indices[train_n:train_n+valid_n],:,:,:], valid_y.shape)
valid_mean = means[rand_indices[train_n:train_n+valid_n],:]
valid_mean = valid_mean.mean()

take = full_x[rand_indices[train_n+valid_n:],:,:,:,:,:]
test_x[:,:,:,:,:] = np.reshape(take, test_x.shape)
test_y[:,:,:] = np.reshape(full_y[rand_indices[train_n+valid_n:],:,:,:], test_y.shape)
test_mean = means[rand_indices[train_n+valid_n:],:]
test_mean = test_mean.mean()

doc = {
    'code': data,
    'dims': dims,
    'batch_size': b_size,
    'train_size': train_size,
    'valid_size': valid_size,
    'test_size': test_size,
    'means': (str(train_mean), str(valid_mean), str(test_mean))
}

doc_code = 'doc' + d_code + '.json'
with open(os.path.join(data_dir, doc_code), 'w') as f:
    json.dump(doc, f)

print 'Training: ', rand_indices[:train_n]
print 'Validation: ', rand_indices[train_n:train_n+valid_n]
print 'Testing: ', rand_indices[train_n+valid_n:]
