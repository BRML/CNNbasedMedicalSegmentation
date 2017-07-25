import numpy as np
import theano
import theano.tensor as T

from breze.arch.construct.base import Layer
from breze.arch.construct.neural import Mlp, FastDropoutMlp
from breze.arch.util import lookup
from breze.arch.component import transfer as _transfer

import basic

standard_conv = 'dnn_conv3d' if theano.config.device.startswith('gpu') else 'conv3d2d'

class SequentialModel(Layer):
    def __init__(self, inpt, image_height,
                 image_width, image_depth,
                 n_channels, out_transfer,
                 layer_vars, using_bn=False,
                 declare=None, name=None):
        '''
        Sequential model containing multiple layers of type
        conv, maxpool, deconv, shortcut, non_linearity, ...
        :param inpt: input
        :param image_height: input image height
        :param image_width: input image width
        :param image_depth: input image depth
        :param n_channels: number of input channels
        :param out_transfer: the output transfer function to be
                             applied after the last layer.
        :param layer_vars: list of dictionaries. Each dictionary
                       specifies the type of layer and all
                       necessary parameters associated with that
                       type of layer.
                       Each layer must provide an output, the
                       output height, width, depth and the number
                       of output channels to succeeding layers.
        See model.SequentialModel for info on the parameters of
        each layer type.
        '''
        self.inpt = inpt
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channels = n_channels
        self.out_transfer = out_transfer
        self.layer_vars = layer_vars
        self.conv_layers = []

        if using_bn:
            self.bn_layers = []
        else:
            self.bn_layers = None

        super(SequentialModel, self).__init__(declare=declare, name=name)

    def _make_layer(self, lv, inpt, height, width, depth, n_chans, index):
        tp = lv['type']

        if tp == 'conv':
            filter_height, filter_width, filter_depth = lv['fs']
            nkerns = lv['nkerns']
            batch_size = None
            transfer = lv['transfer'] if 'transfer' in lv else 'identity'
            bm = lv['bm'] if 'bm' in lv else 'same'
            stride = lv['stride'] if 'stride' in lv else (1, 1, 1)
            implementation = lv['imp'] if 'imp' in lv else standard_conv
            bias = lv['bias'] if 'bias' in lv else False

            layer = basic.Conv3d(
                inpt=inpt, inpt_height=height,
                inpt_width=width, inpt_depth=depth,
                n_inpt=n_chans, filter_height=filter_height,
                filter_width=filter_width, filter_depth=filter_depth,
                n_output=nkerns, transfer=transfer,
                n_samples=batch_size, border_mode=bm, strides=stride,
                declare=self.declare, implementation=implementation,
                use_bias=bias
            )

            self.conv_layers.append(layer)
        elif tp == 'pool':
            pool_height, pool_width, pool_depth = lv['ps']
            transfer = lv['transfer'] if 'transfer' in lv else 'identity'

            layer = basic.MaxPool3d(
                inpt=inpt, inpt_height=height,
                inpt_width=width, inpt_depth=depth,
                pool_height=pool_height, pool_width=pool_width,
                pool_depth=pool_depth, n_output=n_chans,
                transfer=transfer, declare=self.declare
            )
        elif tp == 'deconv':
            filter_height, filter_width, filter_depth = lv['fs']
            nkerns = lv['nkerns']
            batch_size = None
            transfer = lv['transfer'] if 'transfer' in lv else 'identity'
            implementation = lv['imp'] if 'imp' in lv else standard_conv
            bias = lv['bias'] if 'bias' in lv else False
            up_factor = lv['up'] if 'up' in lv else (2, 2, 2)

            layer = basic.Deconv(
                inpt=inpt, inpt_height=height,
                inpt_width=width, inpt_depth=depth,
                n_inpt=n_chans, n_output=nkerns,
                filter_height=filter_height, filter_width=filter_width,
                filter_depth=filter_depth, transfer=transfer,
                n_samples=batch_size, up_factor=up_factor,
                implementation=implementation, bias=bias,
                declare=self.declare
            )

            self.conv_layers.append(layer.conv_layer)
        elif tp == 'shortcut':
            src_i = lv['src']
            dst_i = lv['dst']
            src_layer = self.layers[src_i]
            dst_layer = self.layers[dst_i]

            transfer = lv['transfer'] if 'transfer' in lv else 'identity'
            projection = lv['proj'] if 'proj' in lv else 'zero_pad'
            implementation = lv['imp'] if 'imp' in lv else standard_conv
            mode = lv['mode'] if 'mode' in lv else 'sum'

            layer = basic.Shortcut(
                src_layer=src_layer, dst_layer=dst_layer,
                transfer=transfer, implementation=implementation,
                projection=projection, mode=mode,
                declare=self.declare
            )
        elif tp == 'non_linearity':
            transfer = lv['transfer']
            prelu = transfer == 'prelu'

            layer = basic.NonLinearity(
                inpt=inpt, inpt_height=height,
                inpt_width=width, inpt_depth=depth,
                n_inpt=n_chans, transfer=transfer,
                prelu=prelu, declare=self.declare
            )
        elif tp == 'batch_norm':
            alpha = lv['alpha']
            version = lv['version'] if 'version' in lv else 'new'

            if version == 'new':
                layer = basic.BatchNorm(
                    inpt=inpt, inpt_height=height,
                    inpt_width=width, inpt_depth=depth,
                    n_inpt=n_chans, alpha=alpha,
                    declare=self.declare
                )
            elif version == 'old':
                layer = basic.OldBN(
                    inpt=inpt, inpt_height=height,
                    inpt_width=width, inpt_depth=depth,
                    n_inpt=n_chans, alpha=alpha,
                    declare=self.declare
                )
            elif version == 'faulty':
                layer = basic.BatchNormFaulty(
                    inpt=inpt, inpt_height=height,
                    inpt_width=width, inpt_depth=depth,
                    n_inpt=n_chans, alpha=alpha,
                    declare=self.declare
                )
            else:
                raise ValueError('BN versions are: old, new, faulty')

            self.bn_layers.append(index)
        elif tp == 'concat':
            left_i = lv['left']
            right_i = lv['right']
            mode = lv['mode'] if 'mode' in lv else 'plain'
            nkerns = lv['nkerns'] if 'nkerns' in lv else None

            left = self.layers[left_i]
            right = self.layers[right_i]

            layer = basic.Concatenate(
                layer_left=left, layer_right=right,
                mode=mode, nkerns=nkerns,
                declare=self.declare
            )
        elif tp == 'f-concat':
            left_i = lv['left']
            right_i = lv['right']
            take_left = lv['tl']
            take_right = lv['tr']

            left = self.layers[left_i]
            right = self.layers[right_i]

            layer = basic.FlexConcatenate(
                layer_left=left, layer_right=right,
                take_left=take_left, take_right=take_right,
                declare=self.declare
            )
        elif tp == 'input':
            mode = lv['mode'] if 'mode' in lv else 'same'

            layer = basic.Input(
                inpt=inpt, inpt_height=height,
                inpt_width=width, inpt_depth=depth,
                n_inpt=n_chans, mode=mode, declare=self.declare
            )
        elif tp == 'skip':
            src = lv['src']
            inpt_layer = self.layers[src]

            layer = basic.Skip(
                inpt_layer=inpt_layer, declare=self.declare
            )
        elif tp == 'gate':
            src = lv['src']
            inpt_layer = self.layers[src]
            take = lv['take']

            layer = basic.Gate(
                inpt_layer=inpt_layer, take=take,
                declare=self.declare
            )
        elif tp == 'bint':
            up_factor = lv['up']

            layer = basic.BilinearUpsample3d(
                inpt=inpt, inpt_height=height,
                inpt_width=width, inpt_depth=depth,
                n_inpt=n_chans, up_factor=up_factor,
                declare=self.declare
            )
        else:
            raise NotImplementedError(
                'Layer types supported are: conv, pool, deconv, shortcut, non_linearity, concat, batch_norm'
            )

        return layer

    def _forward(self):
        self.layers = []

        inpt = self.inpt
        height, width, depth, n_chans = (self.image_height, self.image_width,
                                self.image_depth, self.n_channels)
        for i, lv in enumerate(self.layer_vars):
            layer = self._make_layer(lv, inpt, height, width, depth, n_chans, i)
            self.layers.append(layer)

            inpt = layer.get_output()
            height, width, depth = layer.output_height, layer.output_width, layer.output_depth
            n_chans = layer.n_output

        output = self.layers[-1].get_output()
        self.output = self.out_transfer(output)

class FCN(Layer):
    def __init__(self, inpt, image_height, image_width,
                 image_depth, n_channel, n_hiddens_conv,
                 hidden_transfers_conv, n_hiddens_upconv,
                 hidden_transfers_upconv,
                 d_filter_shapes, u_filter_shapes,
                 down_pools, up_pools, out_transfer,
                 b_modes_down='same', b_modes_up='same',
                 strides_down=(1, 1, 1), up_factors= (2, 2, 2),
                 implementation=standard_conv,
                 declare=None, name=None):
        self.inpt = inpt
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_hiddens_conv = n_hiddens_conv
        self.hidden_transfers_conv = hidden_transfers_conv
        self.n_hiddens_conv = n_hiddens_conv
        self.n_hiddens_upconv = n_hiddens_upconv
        self.hidden_transfers_upconv = hidden_transfers_upconv
        self.down_pools = down_pools
        self.up_pools = up_pools
        self.d_filter_shapes = d_filter_shapes
        self.u_filter_shapes = u_filter_shapes
        self.out_transfer = out_transfer
        self.strides_down = strides_down
        self.implementation = implementation
        self.up_factors = up_factors

        self.b_modes_down = b_modes_down
        self.b_modes_up = b_modes_up

        super(FCN, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.d_cnn = Cnn3dFlex(
            inpt=self.inpt, image_height=self.image_height,
            image_width=self.image_width, image_depth=self.image_depth,
            n_channel=self.n_channel, n_hiddens=self.n_hiddens_conv,
            filter_shapes=self.d_filter_shapes, pool_shapes=self.down_pools,
            hidden_transfers=self.hidden_transfers_conv,
            declare=self.declare, name=self.name, border_modes=self.b_modes_down,
            implementation=self.implementation, strides=self.strides_down
        )

        d_out_height = self.d_cnn.layers[-1].output_height
        d_out_width = self.d_cnn.layers[-1].output_width
        d_out_depth = self.d_cnn.layers[-1].output_depth

        self.u_cnn = UpSampleNetwork3d(
            inpt=self.d_cnn.output, image_height=d_out_height,
            image_width=d_out_width, image_depth=d_out_depth,
            n_channel=self.n_hiddens_conv[-1], n_hiddens=self.n_hiddens_upconv,
            filter_shapes=self.u_filter_shapes, pool_shapes=self.up_pools,
            hidden_transfers=self.hidden_transfers_upconv,
            border_modes=self.b_modes_up, declare=self.declare,
            name=self.name, implementation=self.implementation,
            up_factors=self.up_factors
        )

        output = self.u_cnn.output.dimshuffle(0, 3, 4, 1, 2)
        output = T.reshape(output, (-1, self.n_hiddens_upconv[-1]))

        f = lookup(self.out_transfer, _transfer)
        self.output = T.reshape(f(output), (1, -1, self.n_hiddens_upconv[-1]))

class ConvNet3d(Layer):
    def __init__(self, inpt, image_height, image_width,
                 image_depth, n_channel, n_hiddens_conv,
                 filter_shapes, pool_shapes, n_hiddens_full,
                 hidden_transfers_conv, hidden_transfers_full,
                 n_output, out_transfer, border_modes='valid', declare=None,
                 name=None, implementation=standard_conv,dropout=False,
                 p_dropout_inpt=None, p_dropout_hiddens=None):
        """
        Wraps around a flexible 3d cnn and a succeeding stack of fully-connected layers followed by a classifier.
        """
        self.inpt = inpt
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_hiddens_conv = n_hiddens_conv
        self.n_hiddens_full = n_hiddens_full
        self.filter_shapes = filter_shapes
        self.pool_shapes = pool_shapes
        self.hidden_transfers_conv = hidden_transfers_conv
        self.hidden_transfers_full = hidden_transfers_full
        self.n_output = n_output
        self.out_transfer = out_transfer
        self.implementation = implementation

        self.dropout = dropout
        self.p_dropout_inpt = p_dropout_inpt
        self.p_dropout_hiddens = p_dropout_hiddens

        self.border_modes = border_modes


        super(ConvNet3d, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.cnn = Cnn3dFlex(
            inpt=self.inpt, image_height=self.image_height,
            image_width=self.image_width, image_depth=self.image_depth,
            n_channel=self.n_channel, n_hiddens=self.n_hiddens_conv,
            filter_shapes=self.filter_shapes, pool_shapes=self.pool_shapes,
            hidden_transfers=self.hidden_transfers_conv,
            declare=self.declare, border_modes = self.border_modes,
            implementation=self.implementation
        )

        last_cnn_layer = self.cnn.layers[-1]
        n_cnn_outputs = (last_cnn_layer.output_depth *
                         last_cnn_layer.output_height *
                         last_cnn_layer.output_width *
                         last_cnn_layer.n_output)

        mlp_inpt = self.cnn.output.reshape((self.cnn.output.shape[0], -1))

        if not self.dropout:
            self.mlp = Mlp(
                mlp_inpt,
                n_cnn_outputs,
                self.n_hiddens_full,
                self.n_output,
                self.hidden_transfers_full,
                self.out_transfer,
                declare=self.declare
            )

            output = self.mlp.output
        else:
            print '@cnn3d: using dropout'
            self.mlp = FastDropoutMlp(
                mlp_inpt,
                n_cnn_outputs,
                self.n_hiddens_full,
                self.n_output,
                self.hidden_transfers_full,
                self.out_transfer,
                p_dropout_inpt=self.p_dropout_inpt,
                p_dropout_hiddens=self.p_dropout_hiddens,
                declare=self.declare
            )

            output = T.concatenate(self.mlp.outputs, 1)


        self.output = output

class Cnn3dFlex(Layer):
    def __init__(self, inpt, image_height, image_width,
                 image_depth, n_channel, n_hiddens, filter_shapes,
                 pool_shapes, hidden_transfers, batch_size=None, border_modes='valid',
                 strides=(1, 1, 1), declare=None, name=None,
                 implementation=standard_conv):
        """
        Flexible 3d convolutional network. The flexibility is that you can skip pools.
        :param filter_shapes: [(f_height, f_weight, f_depth)*n]
        :param pool_shapes: [((p_height, p_width, p_depth) or 'no_pool')*n]
        """
        self.inpt = inpt
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_hiddens = n_hiddens
        self.filter_shapes = filter_shapes
        self.pool_shapes = pool_shapes
        self.hidden_transfers = hidden_transfers
        self.batch_size = batch_size
        self.implementation = implementation

        if isinstance(border_modes, list):
            assert len(border_modes) == len(n_hiddens)
        else:
            border_modes = [border_modes] * len(n_hiddens)

        if isinstance(strides, list):
            assert len(strides) == len(n_hiddens)
        else:
            strides = [strides] * len(n_hiddens)

        self.border_modes = border_modes
        self.strides = strides

        super(Cnn3dFlex, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.layers = []

        n_inpts = [self.n_channel] + self.n_hiddens[:-1]
        n_outputs = self.n_hiddens
        transfers = self.hidden_transfers
        b_modes = self.border_modes
        strides = self.strides

        inpt = self.inpt
        height, width, depth = (self.image_height, self.image_width,
                                self.image_depth)

        for n, m, fs, ps, t, bm, st in zip(n_inpts, n_outputs, self.filter_shapes,
                                   self.pool_shapes, transfers, b_modes, strides):
            filter_height, filter_width, filter_depth = fs
            if ps == 'no_pool':
                f_ident = t
            else:
                f_ident = 'identity'
            layer = basic.Conv3d(
                inpt, height, width, depth, n, filter_height,
                filter_width, filter_depth, m, f_ident,
                n_samples=self.batch_size, border_mode=bm,
                declare=self.declare, implementation=self.implementation,
                strides=st
            )
            self.layers.append(layer)

            if not (ps == 'no_pool'):
                pool_height, pool_width, pool_depth = ps
                layer = basic.MaxPool3d(
                    layer.output, layer.output_height, layer.output_width,
                    layer.output_depth, pool_height, pool_width, pool_depth,
                    layer.n_output, transfer=t
                )
                self.layers.append(layer)

            inpt = layer.output
            height, width, depth = (layer.output_height, layer.output_width,
                                    layer.output_depth)

        self.output = self.layers[-1].output

class UpSampleNetwork3d(Layer):
    def __init__(self, inpt, image_height, image_width,
                 image_depth, n_channel, n_hiddens, filter_shapes,
                 pool_shapes, hidden_transfers, batch_size=None,
                 border_modes='same', declare=None,
                 name=None, implementation=standard_conv,
                 up_factors=(2, 2, 2)):
        self.inpt = inpt
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_hiddens = n_hiddens
        self.filter_shapes = filter_shapes
        self.pool_shapes = pool_shapes
        self.hidden_transfers = hidden_transfers
        self.batch_size = batch_size
        self.declare = declare
        self.name = name
        self.implementation = implementation

        if isinstance(border_modes, list):
            assert len(border_modes) == len(n_hiddens)
        else:
            border_modes = [border_modes] * len(n_hiddens)
        self.border_modes = border_modes

        if isinstance(up_factors, list):
            assert len(up_factors) == len(n_hiddens)
        else:
            up_factors = [up_factors] * len(n_hiddens)
        self.up_factors = up_factors

        super(UpSampleNetwork3d, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.layers = []

        n_inpts = [self.n_channel] + self.n_hiddens[:-1]
        n_outputs = self.n_hiddens
        transfers = self.hidden_transfers
        b_modes = self.border_modes
        u_fac = self.up_factors

        inpt = self.inpt
        height, width, depth = (self.image_height, self.image_width,
                                self.image_depth)

        for n, m, fs, ps, t, bm, uf in zip(n_inpts, n_outputs, self.filter_shapes,
                                       self.pool_shapes, transfers, b_modes, u_fac):
            filter_height, filter_width, filter_depth = fs
            if ps == 'no_pool':
                f_ident = t
            else:
                f_ident = 'identity'
            layer = basic.Conv3d(
                inpt, height, width, depth, n, filter_height,
                filter_width, filter_depth, m, f_ident,
                n_samples=self.batch_size, border_mode=bm,
                declare=self.declare, implementation=self.implementation
            )
            self.layers.append(layer)

            if ps != 'no_pool':
                pool_height, pool_width, pool_depth = ps
                layer = basic.NearestNeighborsUpsample3d(
                    inpt=layer.output, inpt_height=layer.output_height,
                    inpt_width=layer.output_width, inpt_depth=layer.output_depth,
                    up_factor=(pool_height, pool_width, pool_depth),
                    to_shape=None,
                    transfer=t, declare=self.declare, name=self.name
                )
                self.layers.append(layer)
            inpt = layer.output
            height, width, depth = (layer.output_height, layer.output_width,
                                    layer.output_depth)

        self.output = self.layers[-1].output

class Cnn3d(Layer):
    
    def __init__(self, inpt, image_height, image_width,
                 image_depth, n_channel, n_hiddens, filter_shapes, 
                 pool_shapes, hidden_transfers, batch_size=None,
                 declare=None, name=None, implementation=standard_conv):
        self.inpt = inpt
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_hiddens = n_hiddens
        self.filter_shapes = filter_shapes
        self.pool_shapes = pool_shapes
        self.hidden_transfers = hidden_transfers
        self.batch_size = batch_size
        self.implementation = implementation

        super(Cnn3d, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.layers = []

        n_inpts = [self.n_channel] + self.n_hiddens[:-1]
        n_outputs = self.n_hiddens
        transfers = self.hidden_transfers

        inpt = self.inpt
        height, width, depth = (self.image_height, self.image_width,
                                self.image_depth)

        for n, m, fs, ps, t in zip(n_inpts, n_outputs, self.filter_shapes,
                                   self.pool_shapes, transfers):
            filter_depth, filter_height, filter_width = fs
            layer = basic.Conv3d(
                inpt, height, width, depth, n, filter_height,
                filter_width, filter_depth, m, 'identity',
                n_samples=self.batch_size,
                declare=self.declare, implementation=self.implementation
            )
            self.layers.append(layer)
   
            pool_depth, pool_height, pool_width = ps
            layer = basic.MaxPool3d(
                layer.output, layer.output_height, layer.output_width,
                layer.output_depth, pool_height, pool_width, pool_depth, 
                layer.n_output, transfer=t 
            )
            self.layers.append(layer)

            inpt = layer.output
            height, width, depth = (layer.output_height, layer.output_width,
                                    layer.output_depth)

        self.output = self.layers[-1].output

class Lenet3d(Layer):

    def __init__(self, inpt, image_height, image_width, 
                 image_depth, n_channel, n_hiddens_conv, 
                 filter_shapes, pool_shapes, n_hiddens_full,
                 hidden_transfers_conv, hidden_transfers_full,
                 n_output, out_transfer, declare=None,
                 name=None, implementation=standard_conv, pool=True):
        self.inpt = inpt
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.n_channel = n_channel
        self.n_hiddens_conv = n_hiddens_conv
        self.n_hiddens_full = n_hiddens_full
        self.filter_shapes = filter_shapes
        self.pool_shapes = pool_shapes
        self.hidden_transfers_conv = hidden_transfers_conv
        self.hidden_transfers_full = hidden_transfers_full
        self.n_output = n_output
        self.out_transfer = out_transfer
        self.implementation = implementation
        self.pool = pool

        super(Lenet3d, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.cnn = Cnn3d(
            self.inpt, self.image_height, self.image_width,
            self.image_depth, self.n_channel, self.n_hiddens_conv,
            self.filter_shapes, self.pool_shapes,
            self.hidden_transfers_conv,
            declare=self.declare, implementation=self.implementation
        )

        last_cnn_layer = self.cnn.layers[-1]
        n_cnn_outputs = (last_cnn_layer.output_depth *
                         last_cnn_layer.output_height *
                         last_cnn_layer.output_width *
                         last_cnn_layer.n_output)

        mlp_inpt = self.cnn.output.reshape((self.cnn.output.shape[0], -1))
        self.mlp = Mlp(
            mlp_inpt,
            n_cnn_outputs,
            self.n_hiddens_full,
            self.n_output,
            self.hidden_transfers_full,
            self.out_transfer,
            declare=self.declare
        )

        self.output = self.mlp.output