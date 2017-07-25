import theano
import theano.tensor as T
from theano.tensor.type import TensorType

import numpy as np

from breze.arch.component.varprop import loss as vp_loss
from breze.arch.construct.simple import SupervisedLoss
from breze.arch.util import ParameterSet, lookup
from breze.learn.base import SupervisedModel

import cnn3d
from basic import SupervisedMultiLoss

def tensor5(name=None, dtype=None):
    """
    Returns a symbolic 5D tensor variable.
    """
    if dtype is None:
        dtype = theano.config.floatX

    type = TensorType(dtype, ((False,)*5))
    return type(name)

class SequentialModel(SupervisedModel):
    '''
    Sequential model consisting of multiple layers of different types.
    '''
    def __init__(self, image_height, image_width,
                 image_depth, n_channels, n_output,
                 layer_vars, out_transfer, loss_id=None,
                 loss_layer_def=None, optimizer='adam',
                 batch_size=1, max_iter=1000,
                 using_bn=False, regularize=False,
                 l1=None, l2=None, perform_transform=None,
                 verbose=False):
        '''
        :param n_output: Number of classes
        :param layer_vars: list of dictionaries. Each dictionary specifies
                           the type of a layer and the values of parameters
                           associated with that kind of layer.
                           Possible layers: conv, pool, deconv
                           conv params: fs(:=filter shape), nkerns(:=feature maps),
                                        transfer(:=non-linearity used), bm(:=border-mode),
                                        stride(:=convolution stride), imp(:=theano implementation),
                                        bias(:=True to use biases, False to omit them)
                           pool params: ps(:=pool shape), transfer(:=non-linearity used)
                           deconv params: fs(:=filter shape), nkerns(:=feature maps),
                                        transfer(:=non-linearity used), imp(:=theano implementation),
                                        bias(:=True to use biases, False to omit them),
                                        up(:=upsampling factor)
                           shortcut params: shortcut computes src.output + dst.output
                                            src(:=index of src_layer), dst(:= index of dst_layer),
                                            proj(:=projection to be used if src and dst have different
                                            numbers of feature maps, 'zero_pad' or 'project'),
                                            transfer(:=non_linearity used), imp(:=theano implementation,
                                            only relevant if src and dst have different numbers of
                                            feature maps and proj=='project')
                           non_linearity params: transfer(:=non-linearity used)
                           see: basic.Conv3d, basic.MaxPool3d, basic.Deconv, basic.Shortcut,
                                basic.NonLinearity

        :param out_transfer: output non-linearity, has to be a callable
        :param loss_id: loss function, has to be a callable or the name
                        of a loss function included in breze.arch.component.loss
        :param optimizer: name of an optimizer supported by climin
        :param batch_size: size of an input batch
        :param max_iter: maximum number of training iterations
        '''
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channels = n_channels
        self.n_output = n_output
        self.layer_vars = layer_vars
        self.out_transfer = out_transfer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.using_bn = using_bn
        self.verbose = verbose
        self.regularize = regularize
        if self.regularize:
            if l1 is None and l2 is None:
                raise ValueError('Asked to use regularization but no input for l1 or l2.')
            else:
                self.l1 = l1
                self.l2 = l2

        if loss_layer_def is None:
            if loss_id is not None:
                self.loss_id = loss_id
                self.loss_layer_def = None
            else:
                raise ValueError('Either loss id or loss layer definition has to be specified.')
        else:
            if loss_id is None:
                self.loss_layer_def = loss_layer_def
                self.loss_id = None
            else:
                raise ValueError('loss_id and loss_layer_def can not be used at the same time.')
        self.perform_transform = perform_transform

        self._init_exprs()

        if self.using_bn:
            self.ext_phase = 0
            self.phase_select = self._phase_select
            self.reset_phase = self._reset_phase
        else:
            self.ext_phase = None
            self.phase_select = None
            self.reset_phase = None

    def _make_loss_layer(self, lv, target, declare, imp_weight):
        mode = lv['mode']
        if mode == 'weighted':
            p_weights = lv['weights']
        else:
            p_weights = None
        loss_fun = lv['loss_fun']
        p_indices = lv['predictions']
        transfer = lv['transfer'] if 'transfer' in lv else self.out_transfer

        layers = self.conv_net.layers

        predictions = []
        for p_i in p_indices:
            predictions.append(layers[p_i].get_output())

        self.loss_layer = SupervisedMultiLoss(
            target=target, predictions=predictions,
            loss=loss_fun, mode=mode,
            p_weights=p_weights, imp_weight=imp_weight,
            transfer=transfer, declare=declare
        )

    def _init_exprs(self):
        inpt = tensor5('inpt')
        target = T.tensor3('target')

        parameters = ParameterSet()

        self.conv_net = cnn3d.SequentialModel(
            inpt=inpt, image_height=self.image_height,
            image_width=self.image_width, image_depth=self.image_depth,
            n_channels=self.n_channels, out_transfer=self.out_transfer,
            layer_vars=self.layer_vars, using_bn=self.using_bn,
            declare=parameters.declare
        )

        output = self.conv_net.output

        if self.imp_weight:
            imp_weight = T.matrix('imp_weight')
        else:
            imp_weight = None

        if self.loss_id is not None:
            self.loss_layer = SupervisedLoss(
                target, output, loss=self.loss_id,
                imp_weight=imp_weight, declare=parameters.declare
            )
        else:
            self._make_loss_layer(
                lv=self.loss_layer_def, target=target,
                imp_weight=imp_weight, declare=parameters.declare
            )

        SupervisedModel.__init__(self, inpt=inpt, target=target,
                                 output=output,
                                 loss=self.loss_layer.total,
                                 parameters=parameters)

        self.exprs['imp_weight'] = imp_weight
        if self.regularize:
            self.exprs['true_loss'] = self.exprs['loss'].copy()
            if self.l2 is not None:
                l2_reg = T.sum(T.sqr(self.parameters.flat)) * self.l2 / 2
                self.exprs['loss'] += l2_reg
            if self.l1 is not None:
                l1_reg = T.sum(T.abs_(self.parameters.flat)) * self.l1
                self.exprs['loss'] += l1_reg

    def _phase_select(self, phase_id):
        if phase_id == 'train':
            phase = 0
        elif phase_id == 'valid' or phase_id == 'infer':
            phase = 1
        else:
            raise ValueError('Phases are: train, valid, infer')

        self.ext_phase = phase
        for l_index in self.conv_net.bn_layers:
            self.conv_net.layers[l_index].set_phase(phase)

    def _reset_phase(self):
        self.phase_select(phase_id='train')

    def get_batchnorm_params(self):
        batchnorm_params = []
        for l_index in self.conv_net.bn_layers:
            mean, std = self.conv_net.layers[l_index].submit()
            if not isinstance(mean, np.ndarray):
                mean = mean.as_numpy_array()
                std = std.as_numpy_array()

            mean = np.asarray(mean, dtype='float32')
            std = np.asarray(std, dtype='float32')

            mean_and_std = (mean, std)
            batchnorm_params.append(mean_and_std)

        return batchnorm_params

    def set_batchnorm_params(self, batchnorm_params):
        index = 0
        for l_index in self.conv_net.bn_layers:
            mean, std = batchnorm_params[index]

            self.conv_net.layers[l_index].running_mean.set_value(mean)
            self.conv_net.layers[l_index].running_std.set_value(std)
            index += 1

    def get_params(self):
        layers = self.conv_net.layers
        params = self.parameters

        for i, l in enumerate(layers):
            if hasattr(l, 'weights'):
                w = params[l.weights]
            else:
                w = None
            if hasattr(l, 'bias'):
                b = params[l.bias]
            else:
                b = None
            yield(w, b, i)

    def initialize_xavier_weights(self):
        layers = self.conv_net.layers
        params = self.parameters

        for layer in layers:
            if hasattr(layer, 'weights'):
                w = layer.get_weights()
                fan_in = layer.get_fan_in()
                params[w] = np.random.normal(0., 1., params[w].shape) * np.sqrt(2./fan_in)
            elif hasattr(layer, 'a'):
                a = layer.a
                params[a] = 0.25


class FCN(SupervisedModel):
    def __init__(self, image_height, image_width, image_depth,
                 n_channel, n_output, n_hiddens_conv, down_filter_shapes,
                 hidden_transfers_conv, down_pools, n_hiddens_upconv,
                 up_filter_shapes, hidden_transfers_upconv, up_pools,
                 out_transfer, loss, optimizer='adam',
                 bm_up='same', bm_down='same',
                 batch_size=1, max_iter=1000,
                 strides_d=(1, 1, 1), up_factors=(2, 2, 2),
                 verbose=False, implementation=False):
        assert len(hidden_transfers_conv) == len(n_hiddens_conv)
        assert len(down_filter_shapes) == len(n_hiddens_conv)
        assert len(down_pools) == len(n_hiddens_conv)

        assert len(hidden_transfers_upconv) == len(n_hiddens_upconv)
        assert len(up_filter_shapes) == len(n_hiddens_upconv)
        assert len(up_pools) == len(n_hiddens_upconv)

        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_output = n_output
        self.n_hiddens_conv = n_hiddens_conv
        self.down_filter_shapes = down_filter_shapes
        self.hidden_transfers_conv = hidden_transfers_conv
        self.down_pools = down_pools
        self.n_hiddens_upconv = n_hiddens_upconv
        self.up_filter_shapes = up_filter_shapes
        self.hidden_transfers_upconv = hidden_transfers_upconv
        self.up_pools = up_pools
        self.out_transfer = out_transfer
        self.loss_ident = loss
        self.optimizer = optimizer
        self.bm_down = bm_down
        self.bm_up = bm_up
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.implementation = implementation
        self.strides_d = strides_d
        self.up_factors = up_factors

        self._init_exprs()

    def _init_exprs(self):
        inpt = tensor5('inpt')
        #inpt.tag.test_value = np.zeros((
        #    2, self.image_depth, self.n_channel,
        #    self.image_height, self.image_width
        #))

        target = T.tensor3('target')
        #target.tag.test_value = np.zeros((
        #    2,self.image_depth*self.image_width*self.image_height, self.n_output
        #))

        parameters = ParameterSet()

        self.conv_net = cnn3d.FCN(
            inpt=inpt, image_height=self.image_height,
            image_width=self.image_width, image_depth=self.image_depth,
            n_channel=self.n_channel, n_hiddens_conv=self.n_hiddens_conv,
            hidden_transfers_conv=self.hidden_transfers_conv,
            n_hiddens_upconv=self.n_hiddens_upconv,
            hidden_transfers_upconv=self.hidden_transfers_upconv,
            d_filter_shapes=self.down_filter_shapes,
            u_filter_shapes=self.up_filter_shapes,
            down_pools=self.down_pools,
            up_pools=self.up_pools,
            out_transfer=self.out_transfer,
            b_modes_down=self.bm_down,
            b_modes_up=self.bm_up,
            implementation=self.implementation,
            strides_down=self.strides_d,
            up_factors=self.up_factors,
            declare=parameters.declare
        )

        output = self.conv_net.output

        if self.imp_weight:
            imp_weight = T.matrix('imp_weight')
        else:
            imp_weight = None

        self.loss_layer = SupervisedLoss(
            target, output, loss=self.loss_ident,
            imp_weight=imp_weight, declare=parameters.declare
        )

        SupervisedModel.__init__(self, inpt=inpt, target=target,
                                 output=output,
                                 loss=self.loss_layer.sample_wise.mean(),
                                 parameters=parameters)

        self.exprs['imp_weight'] = imp_weight


class ConvNet3d(SupervisedModel):
    def __init__(self, image_height, image_width, image_depth,
                 n_channel, n_hiddens_conv, filter_shapes, pool_shapes,
                 n_hiddens_full, n_output, hidden_transfers_conv,
                 hidden_transfers_full, out_transfer, loss, optimizer='adam',
                 batch_size=1, max_iter=1000, verbose=False, border_modes='valid',
                 implementation='dnn_conv3d',
                 dropout=False):
        """Flexible Convolutional neural network model

        Some key things to know:
        :param pool_shapes: list of 3-tuples or string-flags. e.g:
                            [(2,2,2), 'no_pool', (2,2,2)]
                            'no_pool' is to skip pooling whenever necessary.

        Future work:
        This model shouldn't actually have fully-connected layers. Rather, it should
        turn fully-connected layers into convolutional layers as follows:
        FC layer that takes (4*4*4)*10 inpt and outputs 1000 neurons will be turned
        into a convolutional layer with 4*4*4 receptive fields outputting on 1000 feature
        maps, thus producing a (1*1*1)*1000 output. If the output of the classification layer
        has 3 neurons (3 classes), then after the conversion you'll get a (1*1*1)*3 output, which
        will be reshaped to 3 neurons afterwards.
        """
        assert len(hidden_transfers_conv) == len(n_hiddens_conv)
        assert len(n_hiddens_conv) == len(filter_shapes)

        assert len(pool_shapes) == len(filter_shapes)
        assert len(hidden_transfers_full) == len(n_hiddens_full)

        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_hiddens_conv = n_hiddens_conv
        self.filter_shapes = filter_shapes
        self.pool_shapes = pool_shapes
        self.n_hiddens_full = n_hiddens_full
        self.n_output = n_output
        self.hidden_transfers_conv = hidden_transfers_conv
        self.hidden_transfers_full = hidden_transfers_full
        self.out_transfer = out_transfer
        self.loss_ident = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.implementation = implementation

        self.dropout = dropout

        self.border_modes = border_modes

        self._init_exprs()

    def _init_exprs(self):
        inpt = tensor5('inpt')
        inpt.tag.test_value = np.zeros((
            2, self.image_depth, self.n_channel,
            self.image_height, self.image_width
        ))

        target = T.matrix('target')
        target.tag.test_value = np.zeros((
            2, self.n_output
        ))

        parameters = ParameterSet()

        if self.dropout:
            self.p_dropout_inpt = .2
            self.p_dropout_hiddens = [.5] * len(self.n_hiddens_full)
        else:
            self.p_dropout_inpt = None
            self.p_dropout_hiddens = None

        self.conv_net = cnn3d.ConvNet3d(
            inpt=inpt, image_height=self.image_height,
            image_width=self.image_width, image_depth=self.image_depth,
            n_channel=self.n_channel, n_hiddens_conv=self.n_hiddens_conv,
            filter_shapes=self.filter_shapes, pool_shapes=self.pool_shapes,
            n_hiddens_full=self.n_hiddens_full,
            hidden_transfers_conv=self.hidden_transfers_conv,
            hidden_transfers_full=self.hidden_transfers_full, n_output=self.n_output,
            out_transfer=self.out_transfer,
            border_modes=self.border_modes,
            declare=parameters.declare,
            implementation=self.implementation,
            dropout=self.dropout, p_dropout_inpt=self.p_dropout_inpt,
            p_dropout_hiddens=self.p_dropout_hiddens
        )

        output = self.conv_net.output

        if self.imp_weight:
            imp_weight = T.matrix('imp_weight')
        else:
            imp_weight = None

        if not self.dropout:
            loss_id = self.loss_ident
        else:
            loss_id = lookup(self.loss_ident, vp_loss)

        self.loss_layer = SupervisedLoss(
            target, output, loss=loss_id,
            imp_weight=imp_weight, declare=parameters.declare
        )

        SupervisedModel.__init__(self, inpt=inpt, target=target,
                                 output=output,
                                 loss=self.loss_layer.total,
                                 parameters=parameters)

        self.exprs['imp_weight'] = imp_weight


class Lenet3d(SupervisedModel):
    
    def __init__(self, image_height, image_width, image_depth,
                 n_channel, n_hiddens_conv, filter_shapes, pool_shapes,
                 n_hiddens_full, n_output, hidden_transfers_conv,
                 hidden_transfers_full, out_transfer, loss, optimizer='adam',
                 batch_size=1, max_iter=1000, verbose=False, implementation='dnn_conv3d',
                 pool=True):
        assert len(hidden_transfers_conv) == len(n_hiddens_conv)
        assert len(n_hiddens_conv) == len(filter_shapes)

        assert len(pool_shapes) == len(filter_shapes)
        assert len(hidden_transfers_full) == len(n_hiddens_full)

        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_hiddens_conv = n_hiddens_conv
        self.n_hiddens_full = n_hiddens_full
        self.filter_shapes = filter_shapes
        self.pool_shapes = pool_shapes
        self.n_output = n_output
        self.hidden_transfers_conv = hidden_transfers_conv
        self.hidden_transfers_full = hidden_transfers_full
        self.out_transfer = out_transfer
        self.loss_ident = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.implementation=implementation
        self.pool = pool

        self._init_exprs()

    def _init_exprs(self):
        inpt = tensor5('inpt')
        inpt.tag.test_value = np.zeros((
            2, self.image_depth, self.n_channel,
            self.image_height, self.image_width
        ))

        target = T.matrix('target')
        target.tag.test_value = np.zeros((
            2, self.n_output
        ))

        parameters = ParameterSet()
       
        self.lenet = cnn3d.Lenet3d(
            inpt, self.image_height,
            self.image_width, self.image_depth,
            self.n_channel, self.n_hiddens_conv,
            self.filter_shapes, self.pool_shapes,
            self.n_hiddens_full, self.hidden_transfers_conv,
            self.hidden_transfers_full, self.n_output,
            self.out_transfer,
            declare=parameters.declare,
            implementation=self.implementation,
            pool=self.pool
        )

        if self.imp_weight:
            imp_weight = T.matrix('imp_weight')
        else:
            imp_weight = None

        self.loss_layer = SupervisedLoss(
            target, self.lenet.output, loss=self.loss_ident,
            imp_weight=imp_weight, declare=parameters.declare
        )

        SupervisedModel.__init__(self, inpt=inpt, target=target,
                                 output=self.lenet.output,
                                 loss=self.loss_layer.total,
                                 parameters=parameters)

        self.exprs['imp_weight'] = imp_weight
