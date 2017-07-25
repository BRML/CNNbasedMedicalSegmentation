import h5py
import numpy as np
import matplotlib.pyplot as plt

def gen_ims(image_stack):
    for image in image_stack:
        #image = np.transpose(image, (1, 0, 2, 3))
        yield image

data = h5py.File('..//data//datasets//brats_fold0.hdf5', 'r')

starting = 3
x = data['test_x'][starting:]

for im in gen_ims(x):
    for _slice in np.arange(0, im.shape[0], 1):
        im_slice = im[_slice]
        fig = plt.figure()

        i = 1
        keys = ['Flair', 'T1', 'T1c', 'T2']
        for modality in im_slice:
            a = fig.add_subplot(2, 2, i)
            plt.imshow(modality, cmap='Greys_r')
            a.set_title(keys[(i-1)])
            plt.axis('off')
            i += 1
        plt.show()
        plt.close()