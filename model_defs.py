"""
This module is for storing different network architectures.
Each architecture is saved as an instance of the class ModelDef,
which is basically just a wrapper around a list of dictionaries
called layer_vars.
layer_vars represents the layers in a neural network. For every layer
we have a dictionary that stores all the hyperparameters of that layer.
Every model has an id. For instance, the one used in the final results
reported in the paper has the id fcn_rffc4. The following code acquires
the definition of fcn_rffc4:

from model_defs import get_model

model_def = get_model('fcn_rffc4')
"""
import ash
import copy
from transformations import random_geometric_transformation, random_geometric_transformation, full_rotation, RandomSectionSelection

class ModelDef(object):
    def __init__(self, alpha, layer_vars,
                 batchnorm, out_transfer,
                 loss_id=None, loss_layer_def=None,
                 size_reduction=0, bn_version='new',
                 regularize=False, l1=None, l2=None,
                 perform_transform=None, trans_for_valid=None,
                 optimizer=('adam', {}), alternate_size=None,
                 initializer='standard'):
        self.alpha = alpha
        self.layer_vars = layer_vars
        self.batchnorm = batchnorm
        self.loss_id = loss_id
        self.loss_layer_def = loss_layer_def
        self.out_transfer = out_transfer
        self.size_reduction = size_reduction
        self.regularize = regularize
        self.l1 = l1
        self.l2 = l2
        self.bn_version = bn_version
        self.perform_transform = perform_transform
        self.trans_for_valid = trans_for_valid
        self.optimizer = optimizer
        self.alternate_size = alternate_size
        self.initializer = initializer
        for lv in self.layer_vars:
            if lv['type'] == 'batch_norm':
                lv['alpha'] = self.alpha
                if self.bn_version is not None:
                    lv['version'] = self.bn_version

        if loss_id is None:
            lf = self.loss_layer_def['loss_fun']
            if isinstance(lf, str):
                self.loss_name = lf
            else:
                self.loss_name = lf.__name__
        else:
            if isinstance(self.loss_id, str):
                self.loss_name = self.loss_id
            else:
                self.loss_name = self.loss_id.__name__

alpha = 0.9
alpha9 = 0.9
alpha1 = 0.1
alpha5 = 0.5

#random_brain_sections_plus = RandomSectionSelection(data_mode='brain', followed_by=random_geometric_transformation)
#random_hand_sections_plus = RandomSectionSelection(data_mode='hand', followed_by=random_geometric_transformation)
#random_brain_sections = RandomSectionSelection(data_mode='brain', followed_by=None)
#random_hand_sections = RandomSectionSelection(data_mode='hand', followed_by=None)
#fix_brain_sections = RandomSectionSelection(data_mode='brain_fix', followed_by=None)

fcn48 = ModelDef(
    alpha = alpha,
    layer_vars = [
        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8}, # 48
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'pool', 'ps': (2, 2, 2)},
        {'type': 'non_linearity', 'transfer': 'prelu'},

        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 24
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'pool', 'ps': (2, 2, 2)},
        {'type': 'non_linearity', 'transfer': 'prelu'},

        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 12
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'pool', 'ps': (2, 2, 2)},
        {'type': 'non_linearity', 'transfer': 'prelu'},

        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 6
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'prelu'},
        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32},
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'shortcut', 'src': 11, 'dst': 16},
        {'type': 'non_linearity', 'transfer': 'prelu'},

        {'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 32, 'up': (2, 2, 2)}, # 12
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'prelu'},

        {'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 16, 'up': (2, 2, 2)}, # 24
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'prelu'},

        {'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 2, 'up': (2, 2, 2)} # 48
    ],
    batchnorm = True,
    loss_id = ash.tanimoto,
    out_transfer = ash.TransFun(ash.tensor_softmax, 2),
    size_reduction = 0,
)

### COLOR MODELS ###

xcnn = ModelDef(
    alpha=alpha5,
    layer_vars=[
        {'i':0, 'type': 'input'},

        {'i':1, 'type': 'gate', 'src': 0, 'take':[0,3]},
        {'i':2, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 4}, # 3
        {'i':3, 'type': 'batch_norm'},
        {'i':4, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':5, 'type': 'pool', 'ps': (2, 2, 2)},

            {'i':6, 'type': 'gate', 'src': 0, 'take':[1,2]},
            {'i':7, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8},
            {'i':8, 'type': 'batch_norm'},
            {'i':9, 'type': 'non_linearity', 'transfer': 'prelu'},
            {'i':10, 'type': 'pool', 'ps': (2, 2, 2)},

        {'i':11, 'type': 'f-concat', 'left': 5, 'right':10, 'tl': 4, 'tr': 4},
        {'i':12, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 10}, # 7
        {'i':13, 'type': 'batch_norm'},
        {'i':14, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':15, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 10}, # 11
        {'i':16, 'type': 'shortcut', 'src': 11, 'dst': 15},
        {'i':17, 'type': 'pool', 'ps': (2, 2, 2)},

            {'i':18, 'type': 'f-concat', 'left': 5, 'right':10, 'tl': 2, 'tr': 6},
            {'i':19, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 12},
            {'i':20, 'type': 'batch_norm'},
            {'i':21, 'type': 'non_linearity', 'transfer': 'prelu'},
            {'i':22, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 12},
            {'i':23, 'type': 'shortcut', 'src': 17, 'dst': 21},
            {'i':24, 'type': 'pool', 'ps': (2, 2, 2)},

        {'i':25, 'type': 'f-concat', 'left': 17, 'right':24, 'tl': 6, 'tr': 6},
        {'i':26, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns':16}, # 19
        {'i':27, 'type': 'batch_norm'},
        {'i':28, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':29, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 27
        {'i':30, 'type': 'shortcut', 'src': 25, 'dst': 29},
        {'i':31, 'type': 'pool', 'ps': (2, 2, 2)},

            {'i':32, 'type': 'f-concat', 'left': 17, 'right':24, 'tl': 4, 'tr': 12},
            {'i':33, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 24},
            {'i':34, 'type': 'batch_norm'},
            {'i':35, 'type': 'non_linearity', 'transfer': 'prelu'},
            {'i':36, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 24},
            {'i':37, 'type': 'shortcut', 'src': 32, 'dst': 36},
            {'i':38, 'type': 'pool', 'ps': (2, 2, 2)},

        {'i':39, 'type': 'f-concat', 'left': 31, 'right':38, 'tl': 12, 'tr': 12},
        {'i':40, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns':24}, # 19
        {'i':41, 'type': 'batch_norm'},
        {'i':42, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':43, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 24}, # 27
        {'i':44, 'type': 'shortcut', 'src': 39, 'dst': 43},

            {'i':45, 'type': 'f-concat', 'left': 31, 'right':38, 'tl': 8, 'tr': 24},
            {'i':46, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32},
            {'i':47, 'type': 'batch_norm'},
            {'i':48, 'type': 'non_linearity', 'transfer': 'prelu'},
            {'i':49, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32},
            {'i':50, 'type': 'shortcut', 'src': 45, 'dst': 49},

        {'i':51, 'type': 'f-concat', 'left': 44, 'right': 50, 'tl': 10, 'tr': 22},
        {'i':52, 'type': 'batch_norm'},
        {'i':53, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':54, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 32, 'up': (2, 2, 2)},
        {'i':55, 'type': 'batch_norm'},
        {'i':56, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':57, 'type': 'f-concat', 'left': 30, 'right': 37, 'tl': 6, 'tr': 18},
        {'i':58, 'type': 'concat', 'left': 56, 'right': 57},
        {'i':59, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 56},
        {'i':60, 'type': 'batch_norm'},
        {'i':61, 'type': 'non_linearity', 'transfer': 'prelu'},

        {'i':62, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 12},
        {'i':63, 'type': 'batch_norm'},
        {'i':64, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':65, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 12, 'up': (2, 2, 2)},
        {'i':66, 'type': 'batch_norm'},
        {'i':67, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':68, 'type': 'f-concat', 'left': 16, 'right': 23, 'tl': 6, 'tr': 10},
        {'i':69, 'type': 'concat', 'left': 67, 'right': 68},
        {'i':70, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 28},
        {'i':71, 'type': 'batch_norm'},
        {'i':72, 'type': 'non_linearity', 'transfer': 'prelu'},

        {'i':73, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 8},
        {'i':74, 'type': 'batch_norm'},
        {'i':75, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':76, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 8, 'up': (2, 2, 2)},
        {'i':77, 'type': 'batch_norm'},
        {'i':78, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':79, 'type': 'f-concat', 'left': 4, 'right': 9, 'tl': 2, 'tr': 6},
        {'i':80, 'type': 'concat', 'left': 78, 'right': 79},
        {'i':81, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16},
        {'i':82, 'type': 'batch_norm'},
        {'i':83, 'type': 'non_linearity', 'transfer': 'prelu'},

        {'i':84, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
    ],
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    bn_version='new',
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate':5e-5})
)

plain = ModelDef(
    alpha = alpha5,
    layer_vars = [
        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8},
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'rectifier'},
        {'type': 'pool', 'ps': (2, 2, 2)},

        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16},
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'rectifier'},
        {'type': 'pool', 'ps': (2, 2, 2)},

        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32},
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'rectifier'},
        {'type': 'pool', 'ps': (2, 2, 2)},

        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64},
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'rectifier'},
        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64},
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'rectifier'},

        {'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 64, 'up': (2, 2, 2)},
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'rectifier'},

        {'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 32, 'up': (2, 2, 2)},
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'rectifier'},

        {'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 16, 'up': (2, 2, 2)},
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'rectifier'},
        {'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5}
    ],
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    bn_version='new',
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate':5e-5})
)

fcn_rffc5 = ModelDef(
    alpha=alpha5,
    layer_vars=[
        {'i':0, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8},                         # 128*128*96*8 - 3
        {'i':1, 'type': 'batch_norm', 'alpha': alpha},
        {'i':2, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':3, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8, 'stride': (2, 2, 2)},    # 64*64*48*8 - 5

        {'i':4, 'type': 'batch_norm', 'alpha': alpha},
        {'i':5, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':6, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16},                        # 64*64*48*16 - 9
        {'i':7, 'type': 'shortcut', 'src': 3, 'dst': 6},
        {'i':8, 'type': 'batch_norm', 'alpha': alpha},
        {'i':9, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':10, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32, 'stride': (2, 2, 2)},   # 32*32*24*32 - 13

        {'i':11, 'type': 'batch_norm', 'alpha': alpha},
        {'i':12, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':13, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 21
        {'i':14, 'type': 'shortcut', 'src': 10, 'dst': 13},
        {'i':15, 'type': 'batch_norm', 'alpha': alpha},
        {'i':16, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':17, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64, 'stride': (2, 2, 2)},   # 16*16*12*64 - 29

        {'i':18, 'type': 'batch_norm', 'alpha': alpha},
        {'i':19, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':20, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64},                        # 16*16*12*64 - 45
        {'i':21, 'type': 'shortcut', 'src': 17, 'dst': 20},
        {'i':22, 'type': 'batch_norm', 'alpha': alpha},
        {'i':23, 'type': 'non_linearity', 'transfer': 'prelu'},                         # 16*16*12*128

        {'i':24, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 32},
        {'i':25, 'type': 'batch_norm', 'alpha': alpha},
        {'i':26, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':27, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 32, 'up': (2, 2, 2)},     # 32*32*24*32 - 53
        {'i':28, 'type': 'batch_norm', 'alpha': alpha},
        {'i':29, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':30, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64},                        # 32*32*24*64 - 61
        {'i':31, 'type': 'batch_norm', 'alpha': alpha},
        {'i':32, 'type': 'non_linearity', 'transfer': 'prelu'},

        {'i':33, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 32},                        # 32*32*24*16
        {'i':34, 'type': 'batch_norm', 'alpha': alpha},
        {'i':35, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':36, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 16, 'up': (2, 2, 2)},     # 64*64*48*16 - 65
        {'i':37, 'type': 'batch_norm', 'alpha': alpha},
        {'i':38, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':39, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32},                        # 64*64*48*32 - 69
        {'i':40, 'type': 'batch_norm', 'alpha': alpha},
        {'i':41, 'type': 'non_linearity', 'transfer': 'prelu'},

        {'i':42, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 16},                         # 64*64*48*8
        {'i':43, 'type': 'batch_norm', 'alpha': alpha},
        {'i':44, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':45, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 8, 'up': (2, 2, 2)}, # 71
        {'i':46, 'type': 'batch_norm', 'alpha': alpha},
        {'i':47, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':48, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 73
        {'i':49, 'type': 'batch_norm', 'alpha': alpha},
        {'i':50, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':51, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},                         # 128*128*96*2

        {'i':52, 'type': 'skip', 'src': 32},
        {'i':53, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':54, 'type': 'bint', 'up': 2},
        {'i':55, 'type': 'skip', 'src': 41},
        {'i':56, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':57, 'type': 'shortcut', 'src': 54, 'dst': 56},
        {'i':58, 'type': 'bint', 'up': 2},
        {'i':59, 'type': 'shortcut', 'src': 51, 'dst': 58}
    ],
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    size_reduction=0,
    bn_version='new',
    regularize=False,
    l1=None, # val like 30171.708
    l2=None,  #0.001 val like 736.552
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate':5e-5})
)

fcn_rffc4 = ModelDef(
    alpha=alpha5,
    layer_vars=[
        {'i':0, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8},                         # 128*128*96*8 - 3
        {'i':1, 'type': 'batch_norm', 'alpha': alpha},
        {'i':2, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':3, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8, 'stride': (2, 2, 2)},    # 64*64*48*8 - 5

        {'i':4, 'type': 'batch_norm', 'alpha': alpha},
        {'i':5, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':6, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16},                        # 64*64*48*16 - 9
        {'i':7, 'type': 'shortcut', 'src': 3, 'dst': 6},
        {'i':8, 'type': 'batch_norm', 'alpha': alpha},
        {'i':9, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':10, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32, 'stride': (2, 2, 2)},   # 32*32*24*32 - 13

        {'i':11, 'type': 'batch_norm', 'alpha': alpha},
        {'i':12, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':13, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 21
        {'i':14, 'type': 'shortcut', 'src': 10, 'dst': 13},
        {'i':15, 'type': 'batch_norm', 'alpha': alpha},
        {'i':16, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':17, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64, 'stride': (2, 2, 2)},   # 16*16*12*64 - 29

        {'i':18, 'type': 'batch_norm', 'alpha': alpha},
        {'i':19, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':20, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64},                        # 16*16*12*64 - 45
        {'i':21, 'type': 'shortcut', 'src': 17, 'dst': 20},
        {'i':22, 'type': 'batch_norm', 'alpha': alpha},
        {'i':23, 'type': 'non_linearity', 'transfer': 'prelu'},                         # 16*16*12*128

        {'i':24, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 32},
        {'i':25, 'type': 'batch_norm', 'alpha': alpha},
        {'i':26, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':27, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 32, 'up': (2, 2, 2)},     # 32*32*24*32 - 53
        {'i':28, 'type': 'batch_norm', 'alpha': alpha},
        {'i':29, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':30, 'type': 'concat', 'left':16, 'right':29},                              # 32*32*24*64
        {'i':31, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64},                        # 32*32*24*64 - 61
        {'i':32, 'type': 'batch_norm', 'alpha': alpha},
        {'i':33, 'type': 'non_linearity', 'transfer': 'prelu'},

        {'i':34, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 32},                        # 32*32*24*16
        {'i':35, 'type': 'batch_norm', 'alpha': alpha},
        {'i':36, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':37, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 16, 'up': (2, 2, 2)},     # 64*64*48*16 - 65
        {'i':38, 'type': 'batch_norm', 'alpha': alpha},
        {'i':39, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':40, 'type': 'concat', 'left': 9, 'right': 39},                             # 64*64*48*32
        {'i':41, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32},                        # 64*64*48*32 - 69
        {'i':42, 'type': 'batch_norm', 'alpha': alpha},
        {'i':43, 'type': 'non_linearity', 'transfer': 'prelu'},

        {'i':44, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 16},                         # 64*64*48*8
        {'i':45, 'type': 'batch_norm', 'alpha': alpha},
        {'i':46, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':47, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 8, 'up': (2, 2, 2)}, # 71
        {'i':48, 'type': 'batch_norm', 'alpha': alpha},
        {'i':49, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':50, 'type': 'concat', 'left': 2, 'right': 49},                             # 128*128*96*16
        {'i':51, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 73
        {'i':52, 'type': 'batch_norm', 'alpha': alpha},
        {'i':53, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':54, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},                         # 128*128*96*2

        {'i':55, 'type': 'skip', 'src': 33},
        {'i':56, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':57, 'type': 'bint', 'up': 2},
        {'i':58, 'type': 'skip', 'src': 43},
        {'i':59, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':60, 'type': 'shortcut', 'src': 57, 'dst': 59},
        {'i':61, 'type': 'bint', 'up': 2},
        {'i':62, 'type': 'shortcut', 'src': 54, 'dst': 61}
    ],
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    size_reduction=0,
    bn_version='new',
    regularize=False,
    l1=None, # val like 30171.708
    l2=None,  #0.001 val like 736.552
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate':5e-5})
)

fcn_rffc3 = ModelDef(
    alpha=alpha5,
    layer_vars=[
        {'i':0, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8},                         # 128*128*96*8 - 3
        {'i':1, 'type': 'batch_norm', 'alpha': alpha},
        {'i':2, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':3, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8, 'stride': (2, 2, 2)},    # 64*64*48*8 - 5

        {'i':4, 'type': 'batch_norm', 'alpha': alpha},
        {'i':5, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':6, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16},                        # 64*64*48*16 - 9
        {'i':7, 'type': 'shortcut', 'src': 3, 'dst': 6},
        {'i':8, 'type': 'batch_norm', 'alpha': alpha},
        {'i':9, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':10, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32, 'stride': (2, 2, 2)},   # 32*32*24*32 - 13

        {'i':11, 'type': 'batch_norm', 'alpha': alpha},
        {'i':12, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':13, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 21
        {'i':14, 'type': 'shortcut', 'src': 10, 'dst': 13},
        {'i':15, 'type': 'batch_norm', 'alpha': alpha},
        {'i':16, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':17, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64, 'stride': (2, 2, 2)},   # 16*16*12*64 - 29

        {'i':18, 'type': 'batch_norm', 'alpha': alpha},
        {'i':19, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':20, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64},                        # 16*16*12*64 - 45
        {'i':21, 'type': 'shortcut', 'src': 17, 'dst': 20},
        {'i':22, 'type': 'batch_norm', 'alpha': alpha},
        {'i':23, 'type': 'non_linearity', 'transfer': 'prelu'},                         # 16*16*12*128

        {'i':24, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 32},
        {'i':25, 'type': 'batch_norm', 'alpha': alpha},
        {'i':26, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':27, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 32, 'up': (2, 2, 2)},     # 32*32*24*32 - 53
        {'i':28, 'type': 'batch_norm', 'alpha': alpha},
        {'i':29, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':30, 'type': 'shortcut', 'src': 16, 'dst': 29},                              # 32*32*24*64
        {'i':31, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64},                        # 32*32*24*64 - 61
        {'i':32, 'type': 'batch_norm', 'alpha': alpha},
        {'i':33, 'type': 'non_linearity', 'transfer': 'prelu'},

        {'i':34, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 32},                        # 32*32*24*16
        {'i':35, 'type': 'batch_norm', 'alpha': alpha},
        {'i':36, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':37, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 16, 'up': (2, 2, 2)},     # 64*64*48*16 - 65
        {'i':38, 'type': 'batch_norm', 'alpha': alpha},
        {'i':39, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':40, 'type': 'shortcut', 'src': 9, 'dst': 39},                             # 64*64*48*32
        {'i':41, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32},                        # 64*64*48*32 - 69
        {'i':42, 'type': 'batch_norm', 'alpha': alpha},
        {'i':43, 'type': 'non_linearity', 'transfer': 'prelu'},

        {'i':44, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 16},                         # 64*64*48*8
        {'i':45, 'type': 'batch_norm', 'alpha': alpha},
        {'i':46, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':47, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 8, 'up': (2, 2, 2)}, # 71
        {'i':48, 'type': 'batch_norm', 'alpha': alpha},
        {'i':49, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':50, 'type': 'shortcut', 'src': 2, 'dst': 49},                             # 128*128*96*16
        {'i':51, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 73
        {'i':52, 'type': 'batch_norm', 'alpha': alpha},
        {'i':53, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':54, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},                         # 128*128*96*2

        {'i':55, 'type': 'skip', 'src': 33},
        {'i':56, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':57, 'type': 'bint', 'up': 2},
        {'i':58, 'type': 'skip', 'src': 43},
        {'i':59, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':60, 'type': 'shortcut', 'src': 57, 'dst': 59},
        {'i':61, 'type': 'bint', 'up': 2},
        {'i':62, 'type': 'shortcut', 'src': 54, 'dst': 61}
    ],
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    size_reduction=0,
    bn_version='new',
    regularize=False,
    l1=None, # val like 30171.708
    l2=None,  #0.001 val like 736.552
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate':5e-5})
)

fcn_rffc3x = ModelDef(
    alpha=fcn_rffc3.alpha,
    layer_vars=copy.deepcopy(fcn_rffc3.layer_vars),
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    bn_version='new',
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate':5e-5}),
    initializer='xavier'
)

fcn_rffc3c = ModelDef(
    alpha=fcn_rffc3.alpha,
    layer_vars=copy.deepcopy(fcn_rffc3.layer_vars),
    batchnorm=True,
    loss_id=ash.fcn_cat_ce,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    bn_version='new',
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate':5e-5}),
)

fcn_rffc = ModelDef(
    alpha=alpha5,
    layer_vars=[
        {'i':0, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8},                         # 128*128*96*8 - 3
        {'i':1, 'type': 'batch_norm', 'alpha': alpha},
        {'i':2, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':3, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8, 'stride': (2, 2, 2)},    # 64*64*48*8 - 5

        {'i':4, 'type': 'batch_norm', 'alpha': alpha},
        {'i':5, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':6, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16},                        # 64*64*48*16 - 9
        {'i':7, 'type': 'shortcut', 'src': 3, 'dst': 6},
        {'i':8, 'type': 'batch_norm', 'alpha': alpha},
        {'i':9, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':10, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32, 'stride': (2, 2, 2)},   # 32*32*24*32 - 13

        {'i':11, 'type': 'batch_norm', 'alpha': alpha},
        {'i':12, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':13, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 21
        {'i':14, 'type': 'shortcut', 'src': 10, 'dst': 13},
        {'i':15, 'type': 'batch_norm', 'alpha': alpha},
        {'i':16, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':17, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64, 'stride': (2, 2, 2)},   # 16*16*12*64 - 29

        {'i':18, 'type': 'batch_norm', 'alpha': alpha},
        {'i':19, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':20, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64},                        # 16*16*12*64 - 45
        {'i':21, 'type': 'shortcut', 'src': 17, 'dst': 20},
        {'i':22, 'type': 'batch_norm', 'alpha': alpha},
        {'i':23, 'type': 'non_linearity', 'transfer': 'prelu'},                         # 16*16*12*128

        {'i':24, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 32},
        {'i':25, 'type': 'batch_norm', 'alpha': alpha},
        {'i':26, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':27, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 32, 'up': (2, 2, 2)},     # 32*32*24*32 - 53
        {'i':28, 'type': 'batch_norm', 'alpha': alpha},
        {'i':29, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':30, 'type': 'concat', 'left':16, 'right':29},                              # 32*32*24*64
        {'i':31, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64},                        # 32*32*24*64 - 61
        {'i':32, 'type': 'shortcut', 'src': 16, 'dst': 31},
        {'i':33, 'type': 'batch_norm', 'alpha': alpha},
        {'i':34, 'type': 'non_linearity', 'transfer': 'prelu'},

        {'i':35, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 32},                        # 32*32*24*16
        {'i':36, 'type': 'batch_norm', 'alpha': alpha},
        {'i':37, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':38, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 16, 'up': (2, 2, 2)},     # 64*64*48*16 - 65
        {'i':39, 'type': 'batch_norm', 'alpha': alpha},
        {'i':40, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':41, 'type': 'concat', 'left': 9, 'right': 40},                             # 64*64*48*32
        {'i':42, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32},                        # 64*64*48*32 - 69
        {'i':43, 'type': 'shortcut', 'src': 9, 'dst': 42},
        {'i':44, 'type': 'batch_norm', 'alpha': alpha},
        {'i':45, 'type': 'non_linearity', 'transfer': 'prelu'},

        {'i':46, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 16},                         # 64*64*48*8
        {'i':47, 'type': 'batch_norm', 'alpha': alpha},
        {'i':48, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':49, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 8, 'up': (2, 2, 2)}, # 71
        {'i':50, 'type': 'batch_norm', 'alpha': alpha},
        {'i':51, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':52, 'type': 'concat', 'left': 2, 'right': 51},                             # 128*128*96*16
        {'i':53, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 73
        {'i':54, 'type': 'shortcut', 'src': 2, 'dst': 53},
        {'i':55, 'type': 'batch_norm', 'alpha': alpha},
        {'i':56, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':57, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},                         # 128*128*96*2

        {'i':58, 'type': 'skip', 'src': 34},
        {'i':59, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':60, 'type': 'bint', 'up': 2},
        {'i':61, 'type': 'skip', 'src': 45},
        {'i':62, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':63, 'type': 'shortcut', 'src': 60, 'dst': 62},
        {'i':64, 'type': 'bint', 'up': 2},
        {'i':65, 'type': 'shortcut', 'src': 57, 'dst': 64}
    ],
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    size_reduction=0,
    bn_version='new',
    regularize=False,
    l1=None, # val like 30171.708
    l2=None,  #0.001 val like 736.552
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate':5e-5})
)

fcn_rffc2 = ModelDef(
    alpha=fcn_rffc.alpha,
    layer_vars=copy.deepcopy(fcn_rffc.layer_vars),
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    bn_version='new',
    regularize=False,
    l1=None,
    l2=None,
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate':5e-6})
)

fcn_rffc_mean = ModelDef(
    alpha=fcn_rffc.alpha,
    layer_vars=copy.deepcopy(fcn_rffc.layer_vars),
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    bn_version='new',
    regularize=False,
    l1=None,
    l2=None,
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate':5e-5})
)

res_fcn_add = ModelDef(
    alpha=alpha5,
    layer_vars=[
        {'i':0, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8},  # 3
        {'i':1, 'type': 'batch_norm'},
        {'i':2, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':3, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8, 'stride': (2, 2, 2)}, # 5

        {'i':4, 'type': 'batch_norm'},
        {'i':5, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':6, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 9
        {'i':7, 'type': 'batch_norm'},
        {'i':8, 'type': 'non_linearity', 'transfer': 'rectifier'}, # 13
        {'i':9, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16},
        {'i':10, 'type': 'shortcut', 'src': 3, 'dst': 9},
        {'i':11, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32, 'stride': (2, 2, 2)}, # 17

        {'i':12, 'type': 'batch_norm'},
        {'i':13, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':14, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 25
        {'i':15, 'type': 'batch_norm'},
        {'i':16, 'type': 'non_linearity', 'transfer': 'rectifier'}, # 33
        {'i':17, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32},
        {'i':18, 'type': 'shortcut', 'src': 11, 'dst': 17},
        {'i':19, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64, 'stride': (2, 2, 2)}, # 41

        {'i':20, 'type': 'batch_norm'},
        {'i':21, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':22, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64}, # 57
        {'i':23, 'type': 'batch_norm'},
        {'i':24, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':25, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64}, # 73
        {'i':26, 'type': 'shortcut', 'src': 19, 'dst': 25},

        {'i':27, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 32},
        {'i':28, 'type': 'batch_norm'},
        {'i':29, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':30, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 32, 'up': (2, 2, 2)}, # 81
        {'i':31, 'type': 'shortcut', 'src': 18, 'dst': 30},
        {'i':32, 'type': 'batch_norm'},
        {'i':33, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':34, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64}, # 89
        {'i':35, 'type': 'batch_norm'},
        {'i':36, 'type': 'non_linearity', 'transfer': 'rectifier'}, # 97
        {'i':37, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64},
        {'i':38, 'type': 'shortcut', 'src': 31, 'dst': 37},

        {'i':39, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 16},
        {'i':40, 'type': 'batch_norm'},
        {'i':41, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':42, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 16, 'up': (2, 2, 2)}, # 101
        {'i':43, 'type': 'shortcut', 'src': 10, 'dst': 42},
        {'i':44, 'type': 'batch_norm'},
        {'i':45, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':46, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 105
        {'i':47, 'type': 'batch_norm'},
        {'i':48, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':49, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 109
        {'i':50, 'type': 'shortcut', 'src': 43, 'dst': 49},

        {'i':51, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 8},
        {'i':52, 'type': 'batch_norm'},
        {'i':53, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':54, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 8, 'up': (2, 2, 2)}, # 111
        {'i':55, 'type': 'shortcut', 'src': 2, 'dst': 54},
        {'i':56, 'type': 'batch_norm'},
        {'i':57, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':58, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 113
        {'i':59, 'type': 'batch_norm'},
        {'i':60, 'type': 'non_linearity', 'transfer': 'rectifier'},
        {'i':61, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 115
        {'i':62, 'type': 'shortcut', 'src': 55, 'dst': 61},

        {'i':63, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},

        {'i':64, 'type': 'skip', 'src': 38},
        {'i':65, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':66, 'type': 'bint', 'up': 2},
        {'i':67, 'type': 'skip', 'src': 50},
        {'i':68, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':69, 'type': 'shortcut', 'src': 66, 'dst': 68},
        {'i':70, 'type': 'bint', 'up': 2},
        {'i':71, 'type': 'shortcut', 'src': 63, 'dst': 70}
    ],
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    size_reduction=0,
    bn_version='new',
    regularize=False,
    l1=None, # val like 30171.708
    l2=None,  #0.001 val like 736.552
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate': 2e-5})
)

res_fcn_concat = ModelDef(
    alpha=alpha5,
    layer_vars=[
        {'i':0, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8}, # 3
        {'i':1, 'type': 'batch_norm'},
        {'i':2, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':3, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 8, 'stride': (2, 2, 2)}, # 5

        {'i':4, 'type': 'batch_norm'},
        {'i':5, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':6, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 9
        {'i':7, 'type': 'batch_norm'},
        {'i':8, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':9, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 13
        {'i':10, 'type': 'shortcut', 'src': 3, 'dst': 9},
        {'i':11, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32, 'stride': (2, 2, 2)}, # 17

        {'i':12, 'type': 'batch_norm'},
        {'i':13, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':14, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 25
        {'i':15, 'type': 'batch_norm'},
        {'i':16, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':17, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 33
        {'i':18, 'type': 'shortcut', 'src': 11, 'dst': 17},
        {'i':19, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64, 'stride': (2, 2, 2)}, # 41

        {'i':20, 'type': 'batch_norm'},
        {'i':21, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':22, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64}, # 57
        {'i':23, 'type': 'batch_norm'},
        {'i':24, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':25, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64}, # 73
        {'i':26, 'type': 'shortcut', 'src': 19, 'dst': 25},

        {'i':27, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 32},
        {'i':28, 'type': 'batch_norm'},
        {'i':29, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':30, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 32, 'up': (2, 2, 2)}, # 81
        {'i':31, 'type': 'concat', 'left': 18, 'right': 30},
        {'i':32, 'type': 'batch_norm'},
        {'i':33, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':34, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64}, # 89
        {'i':35, 'type': 'batch_norm'},
        {'i':36, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':37, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 64}, # 97
        {'i':38, 'type': 'shortcut', 'src': 31, 'dst': 37},

        {'i':39, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 16},
        {'i':40, 'type': 'batch_norm'},
        {'i':41, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':42, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 16, 'up': (2, 2, 2)}, # 101
        {'i':43, 'type': 'concat', 'left': 10, 'right': 42},
        {'i':44, 'type': 'batch_norm'},
        {'i':45, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':46, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 105
        {'i':47, 'type': 'batch_norm'},
        {'i':48, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':49, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 32}, # 109
        {'i':50, 'type': 'shortcut', 'src': 43, 'dst': 49},

        {'i':51, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 8},
        {'i':52, 'type': 'batch_norm'},
        {'i':53, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':54, 'type': 'deconv', 'fs': (3, 3, 3), 'nkerns': 8, 'up': (2, 2, 2)}, # 111
        {'i':55, 'type': 'concat', 'left': 2, 'right': 54},
        {'i':56, 'type': 'batch_norm'},
        {'i':57, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':58, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 113
        {'i':59, 'type': 'batch_norm'},
        {'i':60, 'type': 'non_linearity', 'transfer': 'prelu'},
        {'i':61, 'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 16}, # 115
        {'i':62, 'type': 'shortcut', 'src': 55, 'dst': 61},

        {'i':63, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},

        {'i':64, 'type': 'skip', 'src': 38},
        {'i':65, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':66, 'type': 'bint', 'up': 2},
        {'i':67, 'type': 'skip', 'src': 50},
        {'i':68, 'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5},
        {'i':69, 'type': 'shortcut', 'src': 66, 'dst': 68},
        {'i':70, 'type': 'bint', 'up': 2},
        {'i':71, 'type': 'shortcut', 'src': 63, 'dst': 70}
    ],
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=ash.TransFun(ash.tensor_softmax, 5),
    size_reduction=0,
    bn_version='new',
    regularize=False,
    l1=None, # val like 30171.708
    l2=None,  #0.001 val like 736.552
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate': 5e-5})
)

res_hand_section = ModelDef(
    alpha=res_fcn_add.alpha,
    layer_vars=copy.deepcopy(res_fcn_add.layer_vars),
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=res_fcn_add.out_transfer,
    size_reduction=0,
    bn_version='new',
    regularize=False,
    l1=None,
    l2=None,
    perform_transform=random_geometric_transformation,
    trans_for_valid=random_geometric_transformation,
    optimizer=('adam', {'step_rate': 0.0001, 'decay_mom1': 0.1, 'decay_mom2': 0.001}),
    alternate_size=(144, 120, 96)
)

res_hand_section_concat = ModelDef(
    alpha=res_fcn_concat.alpha,
    layer_vars=copy.deepcopy(res_fcn_concat.layer_vars),
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=res_fcn_concat.out_transfer,
    size_reduction=0,
    bn_version='new',
    regularize=False,
    l1=None,
    l2=None,
    perform_transform=random_geometric_transformation,
    trans_for_valid=random_geometric_transformation,
    optimizer=('adam', {'step_rate': 0.0001, 'decay_mom1': 0.1, 'decay_mom2': 0.001}),
    alternate_size=(144, 120, 96)
)

res_brain_section = ModelDef(
    alpha=res_fcn_add.alpha,
    layer_vars=copy.deepcopy(res_fcn_add.layer_vars),
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=res_fcn_add.out_transfer,
    size_reduction=0,
    bn_version='new',
    regularize=False,
    l1=None,
    l2=None,
    perform_transform=random_geometric_transformation,
    trans_for_valid=random_geometric_transformation,
    optimizer=('adam', {'step_rate': 0.0001, 'decay_mom1': 0.1, 'decay_mom2': 0.001}),
    alternate_size=(80, 72, 64)
)

res_brain_section_concat = ModelDef(
    alpha=res_fcn_concat.alpha,
    layer_vars=copy.deepcopy(res_fcn_concat.layer_vars),
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer=res_fcn_concat.out_transfer,
    size_reduction=0,
    bn_version='new',
    regularize=False,
    l1=None,
    l2=None,
    perform_transform=random_geometric_transformation,
    trans_for_valid=random_geometric_transformation,
    optimizer=('adam', {'step_rate': 0.0001, 'decay_mom1': 0.1, 'decay_mom2': 0.001}),
    alternate_size=(80, 72, 64)
)

ez = ModelDef(
    alpha = alpha5,
    layer_vars = [
        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 5},
        {'type': 'batch_norm', 'alpha': alpha},
        {'type': 'non_linearity', 'transfer': 'prelu'},

        {'type': 'conv', 'fs': (1, 1, 1), 'nkerns': 5}
    ],
    batchnorm=True,
    loss_id=ash.tanimoto,
    out_transfer = ash.TransFun(ash.tensor_softmax, 5),
    size_reduction = 0,
    bn_version='new',
    regularize=False,
    l1=None,
    l2=None,
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate': 5e-5}),
    initializer='xavier'
)

single = ModelDef(
    alpha = alpha,
    layer_vars= [
        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 5}
    ],
    batchnorm=False,
    loss_id=ash.tanimoto,
    out_transfer = ash.TransFun(ash.tensor_softmax, 5),
    size_reduction = 0,
    bn_version = 'new',
    regularize=False,
    l1=None,
    l2=None,
    perform_transform=random_geometric_transformation,
    optimizer=('adam', {'step_rate': 0.0001})
)

single2 = ModelDef(
    alpha = alpha,
    layer_vars= [
        {'type': 'conv', 'fs': (3, 3, 3), 'nkerns': 5}
    ],
    batchnorm=False,
    loss_id=ash.tanimoto,
    out_transfer = ash.TransFun(ash.tensor_softmax, 5),
    size_reduction = 0,
    bn_version = 'new',
    regularize=False,
    l1=None,
    l2=None,
    perform_transform=random_geometric_transformation,
    optimizer= ('adam', {'step_rate': .0002})
)

## TRANSFORM TESTS: ##

res_fcn_add_rot = ModelDef(
    alpha=res_fcn_add.alpha,
    layer_vars=copy.deepcopy(res_fcn_add.layer_vars),
    batchnorm=True,
    loss_id=res_fcn_add.loss_id,
    out_transfer=res_fcn_add.out_transfer,
    size_reduction=res_fcn_add.size_reduction,
    bn_version='new',
    regularize=False,
    l1=None,
    l2=None,
    perform_transform=full_rotation
)

res_fcn_concat_rot = ModelDef(
    alpha=res_fcn_concat.alpha,
    layer_vars=copy.deepcopy(res_fcn_concat.layer_vars),
    batchnorm=True,
    loss_id=res_fcn_concat.loss_id,
    out_transfer=res_fcn_concat.out_transfer,
    size_reduction=res_fcn_concat.size_reduction,
    bn_version='new',
    regularize=False,
    l1=None,
    l2=None,
    perform_transform=full_rotation,
    optimizer=('adam', {'step_rate': 2e-5})
)

ready_models = {
    'ez': ez,
    'single': single,
    'single2': single2,
    'xcnn': xcnn,
    'plain': plain,
    'fcn_rffc': fcn_rffc,
    'fcn_rffc2': fcn_rffc2,
    'fcn_rffc3': fcn_rffc3,
    'fcn_rffc3x': fcn_rffc3x,
    'fcn_rffc3c': fcn_rffc3c,
    'fcn_rffc4': fcn_rffc4,
    'fcn_rffc5': fcn_rffc5,
    'fcn_rffc_mean': fcn_rffc_mean,
    'res_fcn_add': res_fcn_add,
    'res_fcn_concat': res_fcn_concat,
    'res_brain_section': res_brain_section,
    'res_hand_section': res_hand_section,
    'res_hand_section_concat': res_hand_section_concat,
    'res_brain_section_concat': res_brain_section_concat,

    # TRANSFORM TESTS:
    'res_fcn_add_rot': res_fcn_add_rot,
    'res_fcn_concat_rot': res_fcn_concat_rot

    # Regularization Tests:
}

def get_model(model_code):
    return ready_models[model_code]