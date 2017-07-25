# coding: utf-8 
import h5py
import cPickle as pickle
import os
import json

import numpy as np
import gnumpy

import ash
import brain_data_scripts as bds
from model_defs import get_model
from conv3d.model import SequentialModel

def load_mhas_as_dict(path):
    """
    Takes a path to a dictionary holding a series of subdirectories,
    where each subdirectory corresponds to one of the MRI modalities
    'Flair', 'T1', 'T1c' and 'T2'. The subdir corresponding to a mo-
    dality must also contain the name of that modality in its name.
    In other words, we assume the following kind of file hierarchy:
    .
    ├── VSD.Brain_3more.XX.O.OT.54517
    │   ├── License_CC_BY_NC_SA_3.0.txt
    │   └── VSD.Brain_3more.XX.O.OT.54517.mha
    ├── VSD.Brain.XX.O.MR_Flair.54512
    │   ├── License_CC_BY_NC_SA_3.0.txt
    │   └── VSD.Brain.XX.O.MR_Flair.54512.mha
    ├── VSD.Brain.XX.O.MR_T1.54513
    │   ├── License_CC_BY_NC_SA_3.0.txt
    │   └── VSD.Brain.XX.O.MR_T1.54513.mha
    ├── VSD.Brain.XX.O.MR_T1c.54514
    │   ├── License_CC_BY_NC_SA_3.0.txt
    │   └── VSD.Brain.XX.O.MR_T1c.54514.mha
    └── VSD.Brain.XX.O.MR_T2.54515
        ├── License_CC_BY_NC_SA_3.0.txt
        └── VSD.Brain.XX.O.MR_T2.54515.mha
    
    The method will return a tuple dictionary with the modalities
    as keys and numpy ndarrays as values.
    """
    return bds.get_im(path)


def load_dict_as_inputable_ndarray(im):
    """"   
    The method will take a dictionary like the one returned by
    load_mhas_as_dict and return a tuple (image, slices)
    where image is a 5D numpy array with dimensions
    corresponding to: (1, depth, n_chans, height, width)
    and slices will be used to insert the segmentation into 
    a volume of the same size as the original image.

    We need this because the network is trained on images of size
    (128, 160, 144) so during deployment we would ideally extract
    a patch of this size from the original image, segment it and
    re-insert the segmentation into its appropriate place.
    """
    im, slices = bds.get_image_slice(im)

    np_image = bds.get_im_as_ndarray(im, downsize=False)
    np_image = np.transpose(np_image, (1, 0, 2, 3))
    np_image = np_image[np.newaxis]

    return np_image, slices

def build_net(model_folder, model_code, n_classes, train_size, inpt_h, inpt_w, inpt_d, n_channels):
    """
    Takes everything that defines a trained neural network 
    in our setting and returns a function predict that accepts
    a numpy array as input and returns the segmentation corresponding
    to it.
    Parameters:
        model_folder: path to a directory containing the training results
                      of the network.
        model_code:   id of the model. This will be used to find the neural
                      net architecture in model_defs.py
        n_classes:    number of labels in the segmentation problem
        inpt_h:       height of input
        inpt_w:       width of input
        inpt_d:       depth of input
        n_channels:   number of input channels
    """
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

def apply_network(inpt, predict_fn, n_classes=5):
    """Applies the predict function returned by build_net to a numpy array."""
    _, depth, _, height, width = inpt.shape

    model_output = predict_fn(inpt)
    model_output = model_output.as_numpy_array() if isinstance(model_output, gnumpy.garray) else model_output
    fuzzy_seg = np.reshape(
    model_output,
        (height, width, depth, n_classes)
    )
    seg = fuzzy_seg.argmax(axis=3)

    return seg

def segment_dict(im_dict, model_folder, model_code, n_classes=5):
    """
    Segments an image using a trained neural network.
    Parameters:
        im_dict: a dictionary where the keys are 'Flair', 'T1', 'T1c' and 'T2'
                 and the values are numpy ndarrays.
        model_folder: path to a directory containing the training results
                      of the network.
        model_code:   id of the model. This will be used to find the neural
                      net architecture in model_defs.py
    """
    orig_shape = im_dict['Flair'].shape

    inpt, slices = load_dict_as_inputable_ndarray(im_dict)
    train_size, inpt_d, n_channels, inpt_h, inpt_w  = inpt.shape

    predict_fn = build_net(model_folder, model_code, n_classes, train_size, inpt_h, inpt_w, inpt_d, n_channels)

    seg = apply_network(inpt, predict_fn, n_classes)

    segmentation = np.zeros(orig_shape)
    z_s, x_s, y_s = slices
    segmentation[z_s, x_s, y_s] = seg.transpose((2, 0, 1))

    return segmentation

def segment(path, model_folder, model_code, n_classes=5):
    """
    Segments an image using a trained neural network.
    Parameters:
        path: path of a directory containing .mha files in
                its subdirectories. 
                This is specified in: load_mhas_as_dict
        model_folder: path to a directory containing the training results
                      of the network.
        model_code:   id of the model. This will be used to find the neural
                      net architecture in model_defs.py
    """
    im_dict = load_mhas_as_dict(path)
    
    return segment_dict(im_dict, model_folder, model_code, n_classes)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    seg = segment('BRATS2015_Training/HGG/brats_2013_pat0001_1',
                  'dummy45', 'fcn_rffc4')
    for depth_slice in seg:
        bds.vis_col_im(im=np.ones_like(depth_slice), gt=depth_slice)
    