**Code for reproducing the results from [our paper on cnn-based medical image segmentation](https://arxiv.org/abs/1701.03056)**

For any questions or issues, contact Baris via: bkayalibay@gmail.com.

Whenever you use this code, please refer to our publication
```
@article{kayalibay2017cnn,
  title={CNN-based Segmentation of Medical Imaging Data},
  author={Kayalibay, Baris and Jensen, Grady and van der Smagt, Patrick},
  journal={arXiv preprint arXiv:1701.03056},
  year={2017}
}
```

# Requirements:

+ Python 2.7
+ Theano (0.9.0) (along with [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn))
+ [cudamat](https://github.com/cudamat/cudamat)
+ [this fork](https://github.com/bkayalibay/breze) of breze
+ [climin](https://github.com/BRML/climin)
+ h5py
+ SimpleITK

Some of these requirements can be installed with ``pip install package`` (theano, h5py, SimpleITK), 
others (breze, cudamat, climin) should be cloned from the github links provided 
and installed via ``pip install -e .``

# Usage:

To segment new images and get test results, you will need to train the network first. 
The following steps need to be taken to create a data set, train and segment new images:

## Acquire the BRATS 2015 data set:

Go to the [official brats website](http://braintumorsegmentation.org/) and download the 
BRATS 2015 data. Store the **training data** in this directory under a directory called ``BRATS2015_Training``.

## Create a data set:

Run the following line on the terminal:

``python brain_data_scipts/read_images.py``

This will create a .hdf5 file called ``brats_fold0.hdf5`` under ``data/datasets``.
This .hdf5 file contains three randomly created partitions train, valid and test
for training, validation and testing. You can now use it to train a neural network.

## Train the network:

Run:

``python train.py fcn_rffc4 brats_fold0 brats_fold0 600 -ch False``

This will train the network used in our paper on the data set brats_fold0 for
600 iterations over the data set and store the results at the path 
``models/brats_fold0``.

## Test or reuse the trained network:

Once you've trained a network, its parameters and hyperparameters are stored in
a subdirectory of the directory ``models`` (read the docstring of the module 
``train.py`` on how to select this). You can then reuse those parameters using 
the API provided in the module ``segment.py``. In ``segment.py`` you will find 
a function segment that can be used in the following way to segment new images:

``segment('BRATS2015_Training/HGG/brats_2013_pat0001_1', 'results2', 'fcn_rffc4', 5)``

In this example snippet, we are using the network with the id ``fcn_rffc4`` along 
with the parameters stored at ``models/results2`` to segment a medical image 
contained at the path ``BRATS2015_Training/HGG/brats_2013_pat0001_1``.
The general usage is:

``segment([image_path], [params_path], [model_id])``
