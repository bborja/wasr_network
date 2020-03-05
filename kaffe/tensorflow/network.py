import numpy as np
import tensorflow as tf
import scipy.ndimage.filters as fi
import copy
slim = tf.contrib.slim
#from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.keras import layers as keras_ly

DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True, is_training=False, num_classes=21):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup(is_training, num_classes)

    def setup(self, is_training):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            #if isinstance(fed_layer, basestring):  # py 2.7
            if isinstance(fed_layer, str):
                try:
                    #print('Layer ' + fed_layer + ' shape')
                    #print(self.layers[fed_layer].shape)
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def attention_refinment_module(self, input, name):
        #global_pool = tf.reduce_mean(input, [1, 2], keep_dims=True) # tf 1.2
        global_pool = tf.reduce_mean(input, [1, 2], keepdims=True)
        #conv_1 = keras_ly.Conv2D(2048, [1, 1], padding='SAME', name=name+'_conv1')(global_pool)
        conv_1 = keras_ly.Conv2D(input.get_shape()[3], [1, 1], padding='SAME', name=name+'_conv1')(global_pool)
        sigmoid = tf.sigmoid(conv_1, name=name+'_sigmoid')
        mul_out = tf.multiply(input, sigmoid, name=name+'_multiply')

        return mul_out

    @layer
    def attention_refinment_module_new(self, input, name, last_arm=False):
        #global_pool = tf.reduce_mean(input, [1, 2], keep_dims=True) # tf 1.2
        global_pool = tf.reduce_mean(input, [1, 2], keepdims=True)
        conv_1 = keras_ly.Conv2D(input.get_shape()[3], [1, 1], padding='SAME', name=name+'_conv1')(global_pool)
        with tf.variable_scope(name+'_conv1_bn') as scope:
            conv_1_bn = slim.batch_norm(conv_1, fused=True, scope=scope)
        sigmoid = tf.sigmoid(conv_1_bn, name=name+'_sigmoid')
        mul_out = tf.multiply(input, sigmoid, name=name+'_multiply')

        if last_arm:
            #glob_red = tf.reduce_mean(mul_out, [1, 2], keep_dims=True) # tf 1.2
            glob_red = tf.reduce_mean(mul_out, [1, 2], keepdims=True)
            out_scale = tf.multiply(glob_red, mul_out)
            print('last arm shape')
            print(input.shape)
            print(out_scale.shape)
            return out_scale
        else:
            return mul_out

    @layer
    def feature_fusion_module(self, input, name):
        input_big = input[0]
        input_small = input[1]

        up_sampled_input = keras_ly.UpSampling2D(size=(2, 2), name=name+'_upsample')(input_small)

        concat_1 = tf.concat(axis=3, values=[input_big, up_sampled_input], name=name+'_concat')
        conv_1 = keras_ly.Conv2D(1024, [3, 3], padding='SAME', name=name+'_conv1')(concat_1)

        #global_pool = tf.reduce_mean(conv_1, [1, 2], keep_dims=True) # tf 1.2
        global_pool = tf.reduce_mean(conv_1, [1, 2], keepdims=True)
        conv_2 = keras_ly.Conv2D(1024, [1, 1], padding='SAME', name=name+'_conv2')(global_pool)
        conv_3 = keras_ly.Conv2D(1024, [1, 1], padding='SAME', name=name+'_conv3')(conv_2)
        sigmoid = tf.sigmoid(conv_3, name=name+'_sigmoid')

        mul = tf.multiply(sigmoid, conv_1, name=name+'_multiply')
        add_out = tf.add(conv_1, mul, name=name+'_add_out')

        return add_out

    @layer
    def feature_fusion_module_new(self, input, name, num_features):
        input_big = input[0]
        input_small = input[1]

        b_shape = input_big.get_shape()
        s_shape = input_small.get_shape()

        if(b_shape[1].value > s_shape[1].value):
            up_sampled_input = keras_ly.UpSampling2D(size=(2, 2), name=name+'_upsample')(input_small)
        else:
            up_sampled_input = input_small

        concat_1 = tf.concat(axis=3, values=[input_big, up_sampled_input], name=name+'_concat')
        conv_1 = keras_ly.Conv2D(num_features, [3, 3], padding='SAME', name=name+'_conv1')(concat_1)
        conv_1_bn_relu = tf.nn.relu(slim.batch_norm(conv_1, fused=True))

        global_pool = tf.reduce_mean(conv_1_bn_relu, [1, 2], keep_dims=True)

        conv_2 = keras_ly.Conv2D(num_features, [1, 1], padding='SAME', name=name+'_conv2')(global_pool)
        conv_3 = keras_ly.Conv2D(num_features, [1, 1], padding='SAME', name=name+'_conv3')(conv_2)

        sigmoid = tf.sigmoid(conv_3, name=name+'_sigmoid')

        mul = tf.multiply(sigmoid, conv_1_bn_relu, name=name+'_multiply') #sigmoid * conv_1
        add_out = tf.add(conv_1_bn_relu, mul, name=name+'_add_out') # conv_1 + mul

        return add_out

    @layer
    def multiply_two_tensors(self, input, name):
        input_a = input[0]
        input_b = input[1]
        return tf.multiply(input_a, input_b, name=name)

    @layer
    def global_pool(self, input, name, axis):
        #return tf.reduce_mean(input, axis=axis, keep_dims=True, name=name) # tf 1.2
        return tf.reduce_mean(input, axis=axis, keepdims=True, name=name)


    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        print(input.shape)
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            #kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o]) # tf 1.2
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            #kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o]) # tf 1.2
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])

            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output
        
    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def resize_img(self, input, size_x, size_y, name):
        return tf.cast(tf.image.resize_nearest_neighbor(input, size=(size_y, size_x), name=name), dtype=tf.float32)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def max_pool_2(self, input, pool_factor, name='irrelevant'):
        return keras_ly.MaxPooling2D(pool_size=(pool_factor, pool_factor))(input)


    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        print(inputs[0].shape)
        print(inputs[1].shape)
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name)
        
    @layer
    def batch_normalization(self, input, name, is_training, activation_fn=None, scale=True):
        with tf.variable_scope(name) as scope:
            output = slim.batch_norm(
                input,
                activation_fn=activation_fn,
                is_training=is_training,
                updates_collections=None,
                scale=scale,
                scope=scope)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    def se_fully_connected(self, input, units=3, name='sen_fullyconnected'):
        #with tf.name_scope(name):
        return tf.layers.dense(inputs=input, use_bias=True, units=units)

    def se_relu(self, input):
        return tf.nn.relu(input)

    @layer
    def se_sigmoid(self, input, name):
        return tf.nn.sigmoid(input, name=name)

    def se_fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc
			
    @layer
    def global_pool_layer(self, input, axis, name, keep_dims):
        #return tf.reduce_mean(input, axis, keep_dims=keep_dims, name=name) # tf 1.2
        return tf.reduce_mean(input, axis, keepdims=keep_dims, name=name)