import warnings
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv3d2d import conv3d
from theano.tensor.nnet.Conv3D import conv3D
from theano.ifelse import ifelse
from theano.tensor.signal import pool
from breze.arch.component import transfer as _transfer, loss as _loss
from breze.arch.construct.base import Layer
from breze.arch.util import lookup

def prelu(inpt, a):
    '''
    Parametric rectified linear unit, see: https://arxiv.org/pdf/1502.01852.pdf
    '''
    return T.maximum(inpt, 0) + a * T.minimum(inpt, 0)

def elu(inpt):
    '''
    Exponential linear unit, see: arxiv:1511.07289v5 [cs.LG]
    '''
    return (inpt > 0) * inpt  + (inpt <= 0) * (T.exp(inpt) - 1)
    
def tensor_softmax(inpt, n_classes=2):
    output = inpt.dimshuffle(0, 3, 4, 1, 2)
    shuffled_shape = output.shape
    output = T.reshape(output, (-1, n_classes))

    f = lookup('softmax', _transfer)
    output = T.reshape(f(output), shuffled_shape)
    return output.dimshuffle(0, 3, 4, 1, 2)

def stretch_axis(a, axis, factor, original_shape):
    new_shape = [original_shape[0], original_shape[1],
                 original_shape[2], original_shape[3],
                 original_shape[4]]
    new_shape[axis] *= factor
    out_first = T.zeros(new_shape)

    indices_first = [slice(None),] * 5
    indices_first[axis] = slice(0, new_shape[axis], factor*2)
    indices_second = [slice(None),] * 5
    indices_second[axis] = slice(factor*2-1, new_shape[axis], factor*2)

    indices_take_first = [slice(None),] * 5
    indices_take_first[axis] = slice(0, original_shape[axis], factor)
    indices_take_second = [slice(None),] * 5
    indices_take_second[axis] = slice(1, original_shape[axis], factor)

    out_second = T.set_subtensor(out_first[indices_first], a[indices_take_first])
    out = T.set_subtensor(out_second[indices_second], a[indices_take_second])

    return out
    
class Conv3d(Layer):
    def __init__(self, inpt, inpt_height, inpt_width,
                 inpt_depth, n_inpt, filter_height,
                 filter_width, filter_depth, n_output,
                 transfer='identity', n_samples=None,
                 declare=None, name=None, border_mode='valid',
                 implementation='dnn_conv3d', strides=(1, 1, 1),
                 use_bias=False):
        """
        Create one layer of 3d convolution.
        Notes: strides don't work the way they're supposed to with regular
               convolution. First an un-strided convolution takes place
               and then the result is downsized by discarding elements using
               the strides. e.g.: Input size: 48*48*48 -> Conv 3*3*3 -> 46*46*46
                                  -> Stride 2, discards every other voxel -> 23*23*23
                                  Rule of thumb: new_dim = (old_dim - f + 1) / stride
        """
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.inpt_depth = inpt_depth
        self.n_inpt = n_inpt

        self.filter_height = filter_height
        self.filter_width = filter_width
        self.filter_depth = filter_depth

        self.n_output = n_output
        if transfer != 'identity':
            warnings.warn('Transfer functions can only be used in activation layers.', DeprecationWarning)
        self.transfer = 'identity'
        self.n_samples = n_samples

        self.output_height = (inpt_height - filter_height + 1) / strides[0]
        self.output_width = (inpt_width - filter_width + 1) / strides[1]
        self.output_depth = (inpt_depth - filter_depth + 1) / strides[2]

        if border_mode == 'same':
            self.output_height = inpt_height / strides[0]
            self.output_width = inpt_width / strides[1]
            self.output_depth = inpt_depth / strides[2]
            
        if not self.output_height > 0:
            raise ValueError('inpt height smaller than filter height')
        if not self.output_width > 0:
            raise ValueError('inpt width smaller than filter width')
        if not self.output_depth > 0:
            raise ValueError('inpt depth smaller than filter depth')

        self.border_mode = border_mode
        self.implementation = implementation
        self.strides = strides
        self.use_bias = use_bias

        super(Conv3d, self).__init__(declare=declare, name=name)

    def _forward(self):
        inpt = self.inpt

        self.weights = self.declare(
            (self.n_output, self.filter_depth, self.n_inpt,
             self.filter_height, self.filter_width)
        )
        self.bias = self.declare((self.n_output,))

        if self.border_mode == 'same':
            pad_dim1 = self.filter_height - 1
            pad_dim2 = self.filter_width - 1
            pad_dim3 = self.filter_depth - 1

            if pad_dim1 > 0 or pad_dim2 > 0 or pad_dim3 > 0:
                output_shape = (
                    inpt.shape[0], inpt.shape[1] + pad_dim3,
                    inpt.shape[2], inpt.shape[3] + pad_dim1,
                    inpt.shape[4] + pad_dim2
                )
                big_zero = T.zeros(output_shape)
                indices = (
                    slice(None),
                    slice(pad_dim3 // 2, inpt.shape[1] + pad_dim3 // 2),
                    slice(None),
                    slice(pad_dim1 // 2, inpt.shape[3] + pad_dim1 // 2),
                    slice(pad_dim2 // 2, inpt.shape[4] + pad_dim2 // 2)
                )

                inpt = T.set_subtensor(big_zero[indices], inpt)

        #print '@basic.py implementation: ', self.implementation

        if self.implementation == 'conv3d2d':
            self.output_in = conv3d(
                signals=inpt,
                filters=self.weights
            )
            if self.use_bias:
                self.output_in = self.output_in + self.bias.dimshuffle('x', 'x', 0, 'x', 'x')
        elif self.implementation == 'conv3D':
            filters_flip = self.weights[:, ::-1, :, ::-1, ::-1]
            bias = self.bias if self.use_bias else T.zeros(self.bias.shape)
            self.output_in = conv3D(
                V=inpt.dimshuffle(0, 3, 4, 1, 2),
                W=filters_flip.dimshuffle(0, 3, 4, 1, 2),
                b=bias,
                d=(1, 1, 1)
            )
            self.output_in = self.output_in.dimshuffle(0, 3, 4, 1, 2)
        elif self.implementation == 'dnn_conv3d':
            self.output_in = theano.sandbox.cuda.dnn.dnn_conv3d(
                img=inpt.dimshuffle(0, 2, 1, 3, 4),
                kerns=self.weights.dimshuffle(0, 2, 1, 3, 4)
            )
            self.output_in = self.output_in.dimshuffle(0, 2, 1, 3, 4)
            if self.use_bias:
                self.output_in = self.output_in + self.bias.dimshuffle('x', 'x', 0, 'x', 'x')
        else:
            raise NotImplementedError('This class only supports conv3d2d, conv3D and dnn_conv3d')

        self.output = self.output_in

        if self.strides != (1, 1, 1):
            self.output = self.output[:, ::self.strides[2], :, ::self.strides[0], ::self.strides[1]]

    def get_output(self):
        return self.output

    def get_fan_in(self):
        return self.n_inpt * self.filter_height * self.filter_width * self.filter_depth

    def get_weights(self):
        return self.weights


def max_pool_3d(inpt, inpt_shape, ds, ignore_border=True):
    # Downsize 'into the depth' by downsizing twice.
    inpt_shape_4d = (
        inpt_shape[0] * inpt_shape[1],
        inpt_shape[2],
        inpt_shape[3],
        inpt_shape[4]
    )

    inpt_as_tensor4 = T.reshape(inpt, inpt_shape_4d, ndim=4)

    # The first pooling only downsizes the height and the width.
    pool_out1 = pool.pool_2d(inpt_as_tensor4, (ds[1], ds[2]),
                                       ignore_border=True)
    out_shape1 = T.join(0, inpt_shape[:-2], pool_out1.shape[-2:])

    inpt_pooled_once = T.reshape(pool_out1, out_shape1, ndim=5)

    # Shuffle dimensions so the depth is the last dimension.
    inpt_shuffled = inpt_pooled_once.dimshuffle(0, 4, 2, 3, 1)

    shuffled_shape = inpt_shuffled.shape
    # Reshape input to be 4 dimensional.
    shuffle_shape_4d = (
        shuffled_shape[0] * shuffled_shape[1],
        shuffled_shape[2],
        shuffled_shape[3],
        shuffled_shape[4]
    )

    inpt_shuffled_4d = T.reshape(inpt_shuffled, shuffle_shape_4d, ndim=4)

    pool_out2 = pool.pool_2d(inpt_shuffled_4d, (1, ds[0]),
                                       ignore_border=True)
    out_shape2 = T.join(0, shuffled_shape[:-2], pool_out2.shape[-2:])

    inpt_pooled_twice = T.reshape(pool_out2, out_shape2, ndim=5)
    pool_output_fin = inpt_pooled_twice.dimshuffle(0, 4, 2, 3, 1)

    return pool_output_fin


class MaxPool3d(Layer):
    def __init__(self, inpt, inpt_height, inpt_width, inpt_depth,
                 pool_height, pool_width, pool_depth, n_output,
                 transfer='identity', declare=None, name=None):
        """
        One layer of 3D max pooling.
        """

        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.inpt_depth = inpt_depth

        self.pool_height = pool_height
        self.pool_width = pool_width
        self.pool_depth = pool_depth

        if transfer != 'identity':
            warnings.warn('Transfer functions can only be used in activation layers.', DeprecationWarning)
        self.transfer = 'identity'
        self.output_height, _ = divmod(inpt_height, pool_height)
        self.output_width, _ = divmod(inpt_width, pool_width)
        self.output_depth, _ = divmod(inpt_depth, pool_depth)

        if not self.output_height > 0:
            raise ValueError('inpt height smaller than pool height')
        if not self.output_width > 0:
            raise ValueError('inpt width smaller than pool width')
        if not self.output_depth > 0:
            raise ValueError('inpt depth smaller than pool depth')

        self.n_output = n_output

        super(MaxPool3d, self).__init__(declare=declare, name=name)

    def _forward(self):
        poolsize = (self.pool_depth, self.pool_height, self.pool_width)

        self.output = max_pool_3d(
            inpt=self.inpt,
            inpt_shape=self.inpt.shape,
            ds=poolsize,
            ignore_border=True
        )

    def get_output(self):
        return self.output


def upsample_3d(inpt, to_shape, inpt_height, inpt_width, inpt_depth):
    inpt_shape = np.array([inpt_height, inpt_width, inpt_depth])
    to_shape = np.array(to_shape)
    up_factors = to_shape / inpt_shape

    if to_shape[0] >= up_factors[0]*inpt_shape[0]:
        inpt = T.repeat(inpt, up_factors[0], axis=3)
        inpt_shape[0] *= up_factors[0]
    if to_shape[1] >= up_factors[1]*inpt_shape[1]:
        inpt = T.repeat(inpt, up_factors[1], axis=4)
        inpt_shape[1] *= up_factors[1]
    if to_shape[2] >= up_factors[2]*inpt_shape[2]:
        inpt = T.repeat(inpt, up_factors[2], axis=2)
        inpt_shape[2] *= up_factors[2]

    while to_shape[0] >= inpt_shape[0]:
        reps = np.ones(inpt_shape[0], dtype='int16')
        reps[-1] = 2
        inpt = T.repeat(inpt, reps, axis=3)
        inpt_shape[0] += 1
    while to_shape[1] >= inpt_shape[1]:
        reps = np.ones(inpt_shape[1], dtype='int16')
        reps[-1] = 2
        inpt = T.repeat(inpt, reps, axis=4)
        inpt_shape[1] += 1
    while to_shape[2] >= inpt_shape[2]:
        reps = np.ones(inpt_shape[2], dtype='int16')
        reps[-1] = 2
        inpt = T.repeat(inpt, reps, axis=2)
        inpt_shape[2] += 1
    return inpt

def simple_upsample3d(inpt, up_factor):
    inpt = T.repeat(inpt, up_factor[0], axis=3)
    inpt = T.repeat(inpt, up_factor[1], axis=4)
    inpt = T.repeat(inpt, up_factor[2], axis=1)
    #rep = [1, up_factor[2], 1, up_factor[0], up_factor[1]]
    #inpt = T.tile(inpt, rep, ndim=5)
    return inpt

class NearestNeighborsUpsample3d(Layer):
    def __init__(self, inpt, inpt_height, inpt_width,
                 inpt_depth, up_factor=None, to_shape=None,
                 transfer='identity',
                 declare=None, name=None):
        """
        One layer of nearest neighbor upsampling.
        :param inpt: input to be upsampled.
                     Shape: (batch, channel, time, height, width)
        :param to_shape: output shape (3-tuple or list of int).
                         Shape: (height, width, depth)
        """
        self.inpt = inpt
        self.to_shape = to_shape
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.inpt_depth = inpt_depth

        if up_factor is None:
            assert to_shape is not None
            self.output_height = to_shape[0]
            self.output_width = to_shape[1]
            self.output_depth = to_shape[2]
        else:
            assert to_shape is None
            self.output_height = inpt_height * up_factor[0]
            self.output_width = inpt_width * up_factor[1]
            self.output_depth = inpt_depth * up_factor[2]

        self.up_factor = up_factor
        self.to_shape = to_shape

        if transfer != 'identity':
            warnings.warn('Transfer functions can only be used in activation layers.', DeprecationWarning)
        self.transfer = 'identity'

        super(NearestNeighborsUpsample3d, self).__init__(declare=declare, name=name)

    def _forward(self):
        if self.to_shape is not None:
            self.output = upsample_3d(
                self.inpt.dimshuffle(0, 2, 1, 3, 4),
                self.to_shape,
                self.inpt_height,
                self.inpt_width,
                self.inpt_depth
            )
            self.output = self.output.dimshuffle(0, 2, 1, 3, 4)
        else:
            if self.up_factor == (1, 1, 1):
                self.output = self.inpt
            else:
                self.output = simple_upsample3d(self.inpt, self.up_factor)

    def get_output(self):
        return self.output

class BilinearUpsample3d(Layer):
    def __init__(self, inpt, inpt_height, inpt_width,
                 inpt_depth, n_inpt, up_factor=2, declare=None,
                 name=None):
        '''
        Bilinear interpolation through a mild hack.
        This function assumes inpt is: (1, depth, n_inpt, height, width)
        '''
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.inpt_depth = inpt_depth
        self.n_inpt = n_inpt
        self.up_factor= up_factor

        self.output_height = up_factor * inpt_height
        self.output_width = up_factor * inpt_width
        self.output_depth = up_factor * inpt_depth
        self.n_output = n_inpt

        super(BilinearUpsample3d, self).__init__(declare=declare, name=name)

    def _bilinear_upsampling_1D(self, inpt, ratio, batch_size=None, num_input_channels=None):
        '''
        This implementation is a very minimally changed excerpt from:
        https://github.com/Theano/theano/blob/ddfd7d239a1e656cee850cdbc548da63f349c37d/theano/tensor/nnet/abstract_conv.py#L455
        '''
        if theano.config.device.startswith('gpu'):
            from theano.tensor.nnet.abstract_conv import bilinear_kernel_1D, conv2d_grad_wrt_inputs
        else:
            raise AssertionError('Bilinear interpolation requires GPU and cuDNN.')
        try:
            up_bs = batch_size * num_input_channels
        except TypeError:
            up_bs = None
        row, col = inpt.shape[2:]
        up_input = inpt.reshape((-1, 1, row, col))

        concat_mat = T.concatenate((up_input[:, :, :1, :], up_input,
                                    up_input[:, :, -1:, :]), axis=2)

        pad = 2 * ratio - (ratio - 1) // 2 - 1

        kern = bilinear_kernel_1D(ratio=ratio, normalize=True)
        upsampled_row = conv2d_grad_wrt_inputs(
            output_grad=concat_mat,
            filters=kern[np.newaxis, np.newaxis, :, np.newaxis],
            input_shape=(up_bs, 1, row * ratio, col),
            filter_shape=(1, 1, None, 1),
            border_mode=(pad, 0),
            subsample=(ratio, 1),
            filter_flip=True
        )

        return upsampled_row.reshape((inpt.shape[0], inpt.shape[1], row * ratio, col * 1))

    def _forward(self):
        if theano.config.device.startswith('gpu'):
            from theano.tensor.nnet.abstract_conv import bilinear_upsampling
        else:
            raise AssertionError('Bilinear interpolation requires GPU and cuDNN.')

        inpt = T.reshape(self.inpt, (self.inpt_depth, self.n_inpt, self.inpt_height, self.inpt_width))
        pre_res = bilinear_upsampling(input=inpt, ratio=self.up_factor)
        shuffle_res = pre_res.dimshuffle((2, 3, 0, 1))
        res = self._bilinear_upsampling_1D(inpt=shuffle_res, ratio=self.up_factor)
        self.output = res.dimshuffle((2, 3, 0, 1))
        self.output = T.shape_padaxis(self.output, axis=0)
        self.output = T.unbroadcast(self.output, 0)

    def get_output(self):
        return self.output

class Deconv(Layer):
    '''
    Deconvolutional layer.
    Repeats every dimension up_factor[i] times and follows the upsampling by
    a convolution that doesn't change the input size to simulate deconvolution.
    '''
    def __init__(self, inpt, inpt_height, inpt_width,
                 inpt_depth, n_inpt, filter_height,
                 filter_width, filter_depth, n_output,
                 transfer='identity', n_samples=None,
                 up_factor=(2, 2, 2), implementation='dnn_conv3d',
                 bias=False, mode='repeat',
                 declare=None, name=None):
        self.inpt = inpt
        self.inpt_height = inpt_height
        self.inpt_width = inpt_width
        self.inpt_depth = inpt_depth
        self.n_inpt = n_inpt
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.filter_depth = filter_depth
        self.n_output = n_output

        if transfer != 'identity':
            warnings.warn('Transfer functions can only be used in activation layers.', DeprecationWarning)
        self.transfer_id = 'identity'

        self.n_samples = n_samples
        self.up_factor = up_factor
        self.implementation = implementation
        self.bias = bias
        self.mode = mode

        super(Deconv, self).__init__(declare=declare, name=name)

    def sparse_upsample(self):
        up_factor = self.up_factor
        inpt = self.inpt
        new_height = self.inpt_height * up_factor[0]
        new_width = self.inpt_width * up_factor[1]
        new_depth = self.inpt_depth * up_factor[2]

        current_shape = [1, self.inpt_depth, self.n_inpt, self.inpt_height, self.inpt_width]
        inpt_up_first = stretch_axis(inpt, axis=3, factor=up_factor[0], original_shape=current_shape)

        current_shape = [1, self.inpt_depth, self.n_inpt, new_height, self.inpt_width]
        inpt_up_second = stretch_axis(inpt_up_first, axis=4, factor=up_factor[1], original_shape=current_shape)

        current_shape = [1, self.inpt_depth, self.n_inpt, new_height, new_width]
        upsampled_inpt = stretch_axis(inpt_up_second, axis=1, factor=up_factor[2], original_shape=current_shape)

        return upsampled_inpt, new_height, new_width, new_depth

    def _forward(self):
        if self.up_factor != (1, 1, 1) and self.mode == 'repeat':
            self.upsample_layer = NearestNeighborsUpsample3d(
                inpt=self.inpt, inpt_height=self.inpt_height,
                inpt_width=self.inpt_width, inpt_depth=self.inpt_depth,
                up_factor=self.up_factor,
                to_shape=None,
                transfer='identity', declare=self.declare, name=self.name
            )
            inpt = self.upsample_layer.output
            inpt_height = self.upsample_layer.output_height
            inpt_width = self.upsample_layer.output_width
            inpt_depth = self.upsample_layer.output_depth
        elif self.up_factor != (1, 1, 1) and self.mode == 'sparse':
            inpt, inpt_height, inpt_width, inpt_depth = self.sparse_upsample()
        elif self.up_factor != (1, 1, 1) and self.mode == 'bilinear':
            assert(self.up_factor[0] == self.up_factor[1])
            assert(self.up_factor[1] == self.up_factor[2])
            self.upsample_layer = BilinearUpsample3d(
                inpt=self.inpt, inpt_height=self.inpt_height,
                inpt_width=self.inpt_width, inpt_depth=self.inpt_depth,
                n_inpt=self.n_inpt, up_factor=self.up_factor[0],
                declare=self.declare
            )
            inpt = self.upsample_layer.output
            inpt_height = self.upsample_layer.output_height
            inpt_width = self.upsample_layer.output_width
            inpt_depth = self.upsample_layer.output_depth
        elif self.up_factor != (1, 1, 1):
            raise ValueError('Deconv modes are: repeat, sparse, bilinear.')
        else:
            inpt = self.inpt
            inpt_height = self.inpt_height
            inpt_width = self.inpt_width
            inpt_depth = self.inpt_depth

        self.conv_layer = Conv3d(
            inpt=inpt,
            inpt_height=inpt_height,
            inpt_width=inpt_width,
            inpt_depth=inpt_depth,
            n_inpt=self.n_inpt, filter_height=self.filter_height,
            filter_width=self.filter_width, filter_depth=self.filter_depth,
            n_output=self.n_output, transfer=self.transfer_id,
            n_samples=self.n_samples, border_mode='same',
            declare=self.declare, implementation=self.implementation
        )
        # added for Xavier init:
        self.weights = self.conv_layer.weights

        self.output = self.conv_layer.output
        self.output_height = self.conv_layer.output_height
        self.output_width = self.conv_layer.output_width
        self.output_depth = self.conv_layer.output_depth

    def get_output(self):
        return self.output

    def get_weights(self):
        return self.weights

    def get_fan_in(self):
        return self.conv_layer.get_fan_in()


class Shortcut(Layer):
    '''
    Shortcut layer in a residual network as described in:
    http://arxiv.org/pdf/1512.03385v1.pdf (Deep Residual Learning for Image Recognition)
    '''
    def __init__(self, src_layer, dst_layer,
                 transfer='identity', implementation='dnn_conv3d',
                 projection='zero_pad', mode='sum',
                 declare=None, name=None):
        '''
        :param src_layer: layer that produced the input to the
                          stack of layers ending with dst_layer
        :param dst_layer: layer that computes f(x) where x is the
                          output of src_layer
        :param transfer: non-linearity to be applied to the sum f(x) + x
        :param implementation: theano implementation for 3d-convolution.
                               only used if src_layer produces a different
                               number of feature maps than dst_layer and
                               projection is set to 'project'. 'zero_pad'
                               simply adds extra feature maps with zero
                               activations to match the shapes.
        '''
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        if transfer != 'identity':
            warnings.warn('Transfer functions can only be used in activation layers.', DeprecationWarning)
        self.transfer = 'identity'
        self.implementation = implementation
        self.projection = projection

        self.output_height = np.maximum(dst_layer.output_height, src_layer.output_height)
        self.output_width = np.maximum(dst_layer.output_width, src_layer.output_width)
        self.output_depth = np.maximum(dst_layer.output_depth, src_layer.output_depth)
        self.n_output = dst_layer.n_output

        self.mode = mode

        super(Shortcut, self).__init__(declare=declare, name=name)

    def pad_dim(self, left, right, height_diff, width_diff, depth_diff):
        zero_pad = T.zeros_like(right)
        indices = (
            slice(None),
            slice(depth_diff/2, zero_pad.shape[1] - (depth_diff - depth_diff/2)),
            slice(None),
            slice(height_diff/2, zero_pad.shape[3] - (height_diff - height_diff/2)),
            slice(width_diff/2, zero_pad.shape[4] - (width_diff - width_diff/2))
        )

        padded_left = T.set_subtensor(zero_pad[indices], left)
        return padded_left

    def _forward(self):
        inpt_x = self.src_layer.output
        inpt_fx = self.dst_layer.output

        if self.src_layer.n_output != self.dst_layer.n_output:
            if self.projection == 'project':
                proj = Conv3d(
                    inpt=inpt_x, inpt_height=self.src_layer.output_height,
                    inpt_width=self.src_layer.output_width, inpt_depth=self.src_layer.output_depth,
                    n_inpt=self.src_layer.n_output, filter_height=1,
                    filter_width=1, filter_depth=1, n_output=self.dst_layer.n_output,
                    transfer='identity', n_samples=None, border_mode='valid',
                    implementation=self.implementation, strides=(1, 1, 1), use_bias=False,
                    declare=self.declare
                )
                inpt_x = proj.output
            elif self.projection == 'zero_pad':
                projected_shape = (
                    inpt_x.shape[0], inpt_x.shape[1], inpt_fx.shape[2],
                    inpt_x.shape[3], inpt_x.shape[4]
                )
                big_zero = T.zeros(projected_shape)
                indices = (
                    slice(None),
                    slice(None),
                    slice(0, inpt_x.shape[2]),
                    slice(None),
                    slice(None)
                )
                inpt_x = T.set_subtensor(big_zero[indices], inpt_x)
            else:
                raise NotImplementedError('Supported projections: zero_pad, project')

        if self.src_layer.output_height < self.dst_layer.output_height:
            height_diff = self.dst_layer.output_height - self.src_layer.output_height
            assert(self.dst_layer.output_width > self.src_layer.output_width)
            assert(self.dst_layer.output_depth > self.src_layer.output_depth)
            width_diff = self.dst_layer.output_width - self.src_layer.output_width
            depth_diff = self.dst_layer.output_depth - self.src_layer.output_depth
            inpt_x = self.pad_dim(left=inpt_x, right=inpt_fx, height_diff=height_diff, width_diff=width_diff, depth_diff=depth_diff)
        elif self.src_layer.output_height > self.dst_layer.output_height:
            height_diff = self.src_layer.output_height - self.dst_layer.output_height
            assert(self.src_layer.output_width > self.dst_layer.output_width)
            assert (self.src_layer.output_depth > self.dst_layer.output_depth)
            width_diff = self.src_layer.output_width - self.dst_layer.output_width
            depth_diff = self.src_layer.output_depth - self.dst_layer.output_depth
            inpt_fx = self.pad_dim(left=inpt_fx, right=inpt_x, height_diff=height_diff, width_diff=width_diff, depth_diff=depth_diff)

        self.output = inpt_x + inpt_fx
        if self.mode == 'mean':
            self.output /= 2
        elif self.mode != 'sum':
            raise ValueError('Shortcut modes are: sum, mean')

    def get_output(self):
        return self.output

class NonLinearity(Layer):
    '''
    This layer is there to allow for a more atomic architecture where
    non-linearities are handled as layers.
    '''
    def __init__(self, inpt, inpt_height,
                 inpt_width, inpt_depth,
                 n_inpt, transfer, prelu=False,
                 declare=None, name=None):
        self.inpt = inpt
        self.output_height = inpt_height
        self.output_width = inpt_width
        self.output_depth = inpt_depth
        self.n_output = n_inpt
        self.transfer = transfer
        self.prelu = prelu

        super(NonLinearity, self).__init__(declare=declare, name=name)

    def _forward(self):
        if not self.prelu:
            if self.transfer == 't_softmax':
                self.output = tensor_softmax(self.inpt, self.n_output)
            else:
                f = lookup(self.transfer, _transfer)
                self.output = f(self.inpt)
        else:
            self.a = self.declare(
                (1, 1, self.n_output, 1, 1)
            )
            self.output = prelu(self.inpt, self.a)

    def get_output(self):
        return self.output

class BatchNorm(Layer):
    def __init__(self, inpt, inpt_height,
                 inpt_width, inpt_depth,
                 n_inpt, alpha=0.5, training=1,
                 declare=None, name=None):
        '''
        Batch normalization as described in: http://arxiv.org/pdf/1502.03167v3.pdf
        It is assumed that the input has shape (1, depth, n_inpt, height, width) and
        that the normalization is for every feature map.
        :param alpha: Parameter used to compute running metrics(mean and std).
                      The larger alpha is, the higher the influence of recent
                      samples will be.
        '''
        self.inpt = inpt
        self.output_height = inpt_height
        self.output_width = inpt_width
        self.output_depth = inpt_depth
        self.n_output = n_inpt

        self._training = training
        self.eps = 1e-5
        self.alpha = alpha

        super(BatchNorm, self).__init__(declare=declare, name=name)

    @property
    def training(self):
        return self._training

    def set_training(self, bool_val):
        # Convert True/False to 1/0 for Theano
        if bool_val:
            val = 1
        else:
            val = 0
        self._training = val

    def _setup_running_metrics(self, shape):
        self.running_mean = theano.shared(
            np.zeros(shape, dtype='float32'), 'running_mean'
        )
        self.running_std = theano.shared(
            np.ones(shape, dtype='float32'), 'running_std'
        )

    def _forward(self):
        eps = self.eps

        param_size = (1, 1, self.n_output, 1, 1)
        self.gamma = self.declare(param_size)
        self.beta = self.declare(param_size)

        mean = self.inpt.mean(axis=[0, 1, 3, 4], keepdims=False)
        std = self.inpt.std(axis=[0, 1, 3, 4], keepdims=False)

        self._setup_running_metrics(self.n_output)
        self.running_mean.default_update = ifelse(
            self.training,
            (1.0 - self.alpha) * self.running_mean + self.alpha * mean,
            self.running_mean
        )
        self.running_std.default_update = ifelse(
            self.training,
            (1.0 - self.alpha) * self.running_std + self.alpha * std,
            self.running_std
        )

        # This will be optimized away, but ensures the running mean and the running std get updated.
        # Reference: https://gist.github.com/f0k/f1a6bd3c8585c400c190#file-batch_norm-py-L86
        mean += 0 * self.running_mean
        std += 0 * self.running_std

        use_mean = ifelse(self.training, mean, self.running_mean)
        use_std = ifelse(self.training, std, self.running_std)

        use_mean = use_mean.dimshuffle('x', 'x', 0, 'x', 'x')
        use_std = use_std.dimshuffle('x', 'x', 0, 'x', 'x')
        norm_inpt = (self.inpt - use_mean) / (use_std + eps)
        self.output = self.gamma * norm_inpt + self.beta

    def set_phase(self, new_phase):
        # Phase 0: Training, Phase 1: Validation
        if new_phase == 1:
            self.set_training(False)
        elif new_phase == 0:
            self.set_training(True)
        else:
            raise ValueError('Expected 0 or 1, got %i' % new_phase)

    def get_output(self):
        return self.output

    def submit(self):
        return (self.running_mean.get_value(), self.running_std.get_value())

class Concatenate(Layer):
    def __init__(self, layer_left, layer_right, nkerns=None,
                 mode='plain', declare=None, name=None):
        self.layer_left = layer_left
        self.layer_right = layer_right
        self.output_height = layer_left.output_height
        self.output_width = layer_left.output_width
        self.output_depth = layer_left.output_depth
        if mode == 'plain':
            self.n_output = layer_left.n_output + layer_right.n_output
        elif mode == 'truncated':
            if nkerns is None:
                raise ValueError('Have to specify number of features in truncated mode.')
            if nkerns > layer_left.n_output:
                raise ValueError('left does not have %i features.' % nkerns)
            self.n_output = nkerns + layer_right.n_output
        else:
            raise ValueError('Concat modes are: plain, truncated')
        self.mode = mode
        self.nkerns = nkerns

        assert layer_left.output_height == layer_right.output_height
        assert layer_left.output_width == layer_right.output_width
        assert layer_left.output_depth == layer_right.output_depth

        super(Concatenate, self).__init__(declare=declare, name=name)

    def _forward(self):
        left = self.layer_left.get_output()

        if self.mode == 'truncated':
            left = left[:,:,:self.nkerns,:,:]

        right = self.layer_right.get_output()

        self.output = T.concatenate((left, right), axis=2)

    def get_output(self):
        return self.output

class Input(Layer):
    def __init__(self, inpt, inpt_height,
                 inpt_width, inpt_depth,
                 n_inpt, mode='same',
                 declare=None, name=None):
        self.inpt = inpt
        self.output_height = inpt_height
        self.output_width = inpt_width
        self.output_depth = inpt_depth
        self.n_output = n_inpt
        self.mode = mode

        super(Input, self).__init__(declare=declare, name=name)

    def _forward(self):
        if self.mode == 'same':
            self.output = self.inpt
        elif self.mode == 'norm':
            eps = np.array(1e-6, dtype='float32')
            mean = self.inpt.mean(axis=[0, 1, 3, 4], keepdims=True)
            std = self.inpt.std(axis=[0, 1, 3, 4], keepdims=True)
            self.output = (self.inpt - mean) / (std + eps)
        else:
            raise ValueError('Modes are: same, norm')

    def get_output(self):
        return self.output

class Skip(Layer):
    def __init__(self, inpt_layer, declare=None,
                 name=None):
        self.inpt = inpt_layer.get_output()
        self.output_height = inpt_layer.output_height
        self.output_width = inpt_layer.output_width
        self.output_depth = inpt_layer.output_depth
        self.n_output = inpt_layer.n_output

        super(Skip, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.output = self.inpt

    def get_output(self):
        return self.output

class Gate(Layer):
    def __init__(self, inpt_layer, take,
                 declare=None, name=None):
        self.inpt = inpt_layer.get_output()
        self.output_height = inpt_layer.output_height
        self.output_width = inpt_layer.output_width
        self.output_depth = inpt_layer.output_depth

        if not isinstance(take, list):
            take = [take]
        self.n_output = len(take)
        self.take = take

        super(Gate, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.output = self.inpt[:, :, self.take, :, :]

    def get_output(self):
        return self.output

class FlexConcatenate(Layer):
    def __init__(self, layer_left, layer_right,
                 take_left, take_right, declare=None,
                 name=None):
        self.layer_left = layer_left
        self.layer_right = layer_right
        self.output_height = layer_left.output_height
        self.output_width = layer_left.output_width
        self.output_depth = layer_left.output_depth
        self.n_output = take_left + take_right
        self.take_left = take_left
        self.take_right = take_right

        assert layer_left.output_height == layer_right.output_height
        assert layer_left.output_width == layer_right.output_width
        assert layer_left.output_depth == layer_right.output_depth

        super(FlexConcatenate, self).__init__(declare=declare, name=name)

    def _forward(self):
        left = self.layer_left.get_output()
        right = self.layer_right.get_output()

        if self.layer_left.n_output > self.take_left:
            proj_left = Conv3d(
                inpt=left, inpt_height=self.layer_left.output_height,
                inpt_width=self.layer_left.output_width, inpt_depth=self.layer_left.output_depth,
                n_inpt=self.layer_left.n_output, filter_height=1,
                filter_width=1, filter_depth=1, n_output=self.take_left,
                transfer='identity', n_samples=None, strides=(1, 1, 1),
                declare=self.declare
            )
            left = proj_left.output

        if self.layer_right.n_output > self.take_right:
            proj_right = Conv3d(
                inpt=right, inpt_height=self.layer_right.output_height,
                inpt_width=self.layer_right.output_width, inpt_depth=self.layer_right.output_depth,
                n_inpt=self.layer_right.n_output, filter_height=1,
                filter_width=1, filter_depth=1, n_output=self.take_right,
                transfer='identity', n_samples=None, strides=(1, 1, 1),
                declare=self.declare
            )
            right = proj_right.output

        self.output = T.concatenate((left, right), axis=2)

    def get_output(self):
        return self.output
        
class OldBN(Layer):
    def __init__(self, inpt, inpt_height,
                 inpt_width, inpt_depth,
                 n_inpt, alpha=0.5, warn_phase=False,
                 declare=None, name=None):
        '''
        Batch normalization as described in: http://arxiv.org/pdf/1502.03167v3.pdf
        It is assumed that the input has shape (1, depth, n_inpt, height, width) and
        that the normalization is for every feature map.
        :param alpha: Parameter used to compute running metrics(mean and std).
                      The larger alpha is, the higher the influence of recent
                      samples will be.
        '''
        self.inpt = inpt
        self.output_height = inpt_height
        self.output_width = inpt_width
        self.output_depth = inpt_depth
        self.n_output = n_inpt
        self.warn_phase = warn_phase
        self.alpha = alpha
        self.running_mean = theano.shared(
            value=np.zeros((1, 1, self.n_output, 1, 1), dtype='float32')
        )
        self.running_std = theano.shared(
            value=np.ones((1, 1, self.n_output, 1, 1), dtype='float32')
        )

        super(OldBN, self).__init__(declare=declare, name=name)

    def _forward(self):
        param_size = (1, 1, self.n_output, 1, 1)
        self.gamma = self.declare(param_size)
        self.beta = self.declare(param_size)
        eps = 1e-6

        mean = self.inpt.mean(axis=[0, 1, 3, 4], keepdims=True)
        std = self.inpt.std(axis=[0, 1, 3, 4], keepdims=True)

        norm_inpt = (self.inpt - mean) / (std + eps)
        self.output = self.gamma * norm_inpt + self.beta

    def set_phase(self, new_phase):
        if self.warn_phase:
            warnings.warn('OldBN does not have phases.')

    def get_output(self):
        return self.output

    def submit(self):
        return (self.running_mean.get_value(), self.running_std.get_value())

class BatchNormFaulty(Layer):
    def __init__(self, inpt, inpt_height,
                 inpt_width, inpt_depth,
                 n_inpt, alpha=0.1,
                 declare=None, name=None):
        '''
        Batch normalization as described in: http://arxiv.org/pdf/1502.03167v3.pdf
        It is assumed that the input has shape (1, depth, n_inpt, height, width) and
        that the normalization is for every feature map.
        :param alpha: Parameter used to compute running metrics(mean and std).
                      The larger alpha is, the higher the influence of recent
                      samples will be.
        '''
        self.inpt = inpt
        self.output_height = inpt_height
        self.output_width = inpt_width
        self.output_depth = inpt_depth
        self.n_output = n_inpt

        self.freeze_mean = T.zeros((1, 1, self.n_output, 1, 1), dtype='float32')
        self.freeze_std = T.zeros((1, 1, self.n_output, 1, 1), dtype='float32')

        self.phase = theano.shared(
            value=np.array(0, dtype='int8')
        )
        self.alpha = theano.shared(
            value=np.array(1., dtype='float32')
        )
        self.real_alpha = theano.shared(
            value=np.array(alpha, dtype='float32')
        )
        self.alpha.default_update = self.real_alpha

        super(BatchNormFaulty, self).__init__(declare=declare, name=name)

    def _forward(self):
        self.running_mean = theano.shared(
            value=np.zeros((1, 1, self.n_output, 1, 1), dtype='float32')
        )
        self.running_std = theano.shared(
            value=np.zeros((1, 1, self.n_output, 1, 1), dtype='float32')
        )

        param_size = (1, 1, self.n_output, 1, 1)
        self.gamma = self.declare(param_size)
        self.beta = self.declare(param_size)
        eps = np.array(1e-6, dtype='float32')

        mean = T.mean(self.inpt, axis=[0, 1, 3, 4], keepdims=True, dtype='float32')
        std = T.mean(T.sqr(self.inpt - mean), axis=[0, 1, 3, 4], keepdims=True, dtype='float32')
        std = T.sqrt(std + eps)

        self.running_mean.default_update = (
            (1.0 - self.alpha) * self.running_mean + self.alpha * mean
        )
        self.running_std.default_update = (
            (1.0 - self.alpha) * self.running_std + self.alpha * std
        )

        # This will be optimized away, but ensures the running mean and the running std get updated.
        # Reference: https://gist.github.com/f0k/f1a6bd3c8585c400c190#file-batch_norm-py-L86
        mean += 0 * self.running_mean
        std += 0 * self.running_std

        phase = self.phase.get_value()
        # train
        if phase == 0:
            use_mean = mean
            use_std = std
        # valid
        elif phase == 1:
            use_mean = self.freeze_mean
            use_std = self.freeze_std
        # infer
        else:
            use_mean = self.freeze_mean
            use_std = self.freeze_std

        norm_inpt = (self.inpt - use_mean) / use_std
        self.output = self.gamma * norm_inpt + self.beta

    def set_phase(self, new_phase):
        # Completely ignore the phase
        # get the running metrics before validation/inference
        #if phase == 1 or phase == 2:
        #    self.freeze_mean.set_value(self.running_mean.get_value())
        #    self.freeze_std.set_value(self.running_std.get_value())
        # reset the running metrics to old values before continuing the training
        #else:
        #    self.running_mean.set_value(self.freeze_mean)
        #    self.running_std.set_value(self.freeze_std)
        self.phase.set_value(new_phase)

    def get_output(self):
        return self.output

    def submit(self):
        return (self.running_mean.get_value(), self.running_std.get_value())

class SupervisedMultiLoss(Layer):
    def __init__(self, target, predictions, loss, transfer,
                 comp_dim=1, imp_weight=None, p_weights=None,
                 mode='sum', declare=None, name=None):
        '''
        Compute a loss that is a compound of different losses.
        :param target: labels correponding to data
        :param predictions: list of symbolic tensors of the same shape as target
        :param loss: loss function
        '''
        self.target = target
        self.predictions = predictions
        self.loss_ident = loss
        self.imp_weight = imp_weight
        self.comp_dim = comp_dim
        self.mode = mode
        self.transfer = transfer

        if p_weights is not None:
            if self.mode != 'weighted':
                raise ValueError('Set weights for sum but mode is not weighted.')
        else:
            p_weights = [1] * len(predictions)
            if self.mode not in ['sum', 'mean']:
                raise ValueError('No weight scheme is given but mode is not sum or mean.')
        self.p_weights = p_weights

        super(SupervisedMultiLoss, self).__init__(declare=declare, name=name)

    def _forward(self):
        f_loss = lookup(self.loss_ident, _loss)

        self.coord_wise_multi = [f_loss(self.target, self.transfer(pred)) for pred in self.predictions]
        if self.imp_weight is not None:
            self.coord_wise_multi = [coord_wise * self.imp_weight for coord_wise in self.coord_wise_multi]

        self.sample_wise_multi = [coord_wise.sum(self.comp_dim) for coord_wise in self.coord_wise_multi]
        self.total_multi = [sample_wise.mean() for sample_wise in self.sample_wise_multi]

        self.total = T.zeros(self.total_multi[0].shape)
        for tot, pw in zip(self.total_multi, self.p_weights):
            self.total += tot * pw

        if self.mode == 'mean':
            self.total /= len(self.predictions)