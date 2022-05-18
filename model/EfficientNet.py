from tensorflow import keras
from tensorflow.python import tf2
from tensorflow.keras.layers import Input
from functools import wraps, reduce
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Concatenate, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D, ZeroPadding2D, GlobalAveragePooling2D, Reshape, multiply, add, Dense, \
    Dropout, Lambda
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.utils import get_source_inputs, get_file
from tensorflow.keras.regularizers import l2
from keras_applications.imagenet_utils import _obtain_input_shape
import math
from .models import load_custom_model

BASE_WEIGHTS_PATH = (
    'https://github.com/Callidior/keras-applications/'
    'releases/download/efficientnet/')
WEIGHTS_HASHES = {
    'b1': ('8f83b9aecab222a9a2480219843049a1',
           'b20160ab7b79b7a92897fcb33d52cc61')
}

DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def swish(x):
    if K.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return K.tf.nn.swish(x)
        except AttributeError:
            pass

    return x * K.sigmoid(x)


def correct_pad(backend, inputs, kernel_size):
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def block(inputs, activation_fn=swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = Conv2D(filters, 1,
                   padding='same',
                   use_bias=False,
                   kernel_initializer=CONV_KERNEL_INITIALIZER,
                   name=name + 'expand_conv')(inputs)
        x = BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = ZeroPadding2D(padding=correct_pad(K, x, kernel_size),
                          name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = DepthwiseConv2D(kernel_size,
                        strides=strides,
                        padding=conv_pad,
                        use_bias=False,
                        depthwise_initializer=CONV_KERNEL_INITIALIZER,
                        name=name + 'dwconv')(x)
    x = BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = Activation(activation_fn, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        se = Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = Conv2D(filters_se, 1,
                    padding='same',
                    activation=activation_fn,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    name=name + 'se_reduce')(se)
        se = Conv2D(filters, 1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    name=name + 'se_expand')(se)
        if K.backend() == 'theano':
            se = Lambda(
                lambda x: K.pattern_broadcast(x, [True, True, True, False]),
                output_shape=lambda input_shape: input_shape,
                name=name + 'se_broadcast')(se)
        x = multiply([x, se], name=name + 'se_excite')

    x = Conv2D(filters_out, 1,
               padding='same',
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name=name + 'project_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            if tf2.enabled():
                x = Dropout(drop_rate,
                            noise_shape=(None, 1, 1, 1),
                            name=name + 'drop')(x)
            else:
                x = Dropout(drop_rate,
                            # noise_shape=(None, 1, 1, 1),
                            name=name + 'drop')(x)
        x = add([x, inputs], name=name + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation_fn=swish,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    def round_filters(filters, divisor=depth_divisor):
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        return int(math.ceil(depth_coefficient * repeats))

    x = img_input
    x = ZeroPadding2D(padding=correct_pad(K, x, 3),
                      name='stem_conv_pad')(x)
    x = Conv2D(round_filters(32), 3,
               strides=2,
               padding='valid',
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name='stem_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = Activation(activation_fn, name='stem_activation')(x)

    from copy import deepcopy
    blocks_args = deepcopy(blocks_args)

    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, activation_fn, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1

    x = Conv2D(round_filters(1280), 1,
               padding='same',
               use_bias=False,
               kernel_initializer=CONV_KERNEL_INITIALIZER,
               name='top_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = Activation(activation_fn, name='top_activation')(x)
    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name='top_dropout')(x)
        x = Dense(classes,
                  activation='softmax',
                  kernel_initializer=DENSE_KERNEL_INITIALIZER,
                  name='probs')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name=model_name)

    # Load weights.
    file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
    file_name = model_name + file_suff
    weights_path = get_file(file_name,
                            BASE_WEIGHTS_PATH + file_name,
                            cache_subdir='models',
                            file_hash=file_hash)
    model.load_weights(weights_path)
    return model


def EfficientNetB1(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.0, 1.1, 240, 0.2,
                        model_name='efficientnet-b1',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


@wraps(DepthwiseConv2D)
def DarknetDepthwiseConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return DepthwiseConv2D(*args, **darknet_conv_kwargs)


def Darknet_Depthwise_Separable_Conv2D_BN_Leaky(filters, kernel_size=(3, 3), block_id_str=None, **kwargs):
    if not block_id_str:
        block_id_str = str(K.get_uid())
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetDepthwiseConv2D(kernel_size, name='conv_dw_' + block_id_str, **no_bias_kwargs),
        BatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_dw_%s_leaky_relu' % block_id_str),
        Conv2D(filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        BatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_pw_%s_leaky_relu' % block_id_str))


def Depthwise_Separable_Conv2D_BN_Leaky(filters, kernel_size=(3, 3), block_id_str=None):
    if not block_id_str:
        block_id_str = str(K.get_uid())
    return compose(
        DepthwiseConv2D(kernel_size, padding='same', name='conv_dw_' + block_id_str),
        BatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_dw_%s_leaky_relu' % block_id_str),
        Conv2D(filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%s' % block_id_str),
        BatchNormalization(name='conv_pw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_pw_%s_leaky_relu' % block_id_str))


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def mish(x):
    return x * K.tanh(K.softplus(x))


def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Activation(mish))


def Spp_Conv2D_BN_Leaky(x, num_filters):
    y1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    y2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    y3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)

    y = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))([y1, y2, y3, x])
    return y


def make_yolo_head(x, num_filters):
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    return x


def make_yolo_spp_head(x, num_filters):
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    x = Spp_Conv2D_BN_Leaky(x, num_filters)

    x = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    return x


def make_yolo_depthwise_separable_head(x, num_filters, block_id_str=None):
    if not block_id_str:
        block_id_str = str(K.get_uid())
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters * 2, kernel_size=(3, 3),
                                            block_id_str=block_id_str + '_1'),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters * 2, kernel_size=(3, 3),
                                            block_id_str=block_id_str + '_2'),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    return x


def make_yolo_spp_depthwise_separable_head(x, num_filters, block_id_str=None):
    if not block_id_str:
        block_id_str = str(K.get_uid())
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters * 2, kernel_size=(3, 3),
                                            block_id_str=block_id_str + '_1'),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    x = Spp_Conv2D_BN_Leaky(x, num_filters)

    x = compose(
        Depthwise_Separable_Conv2D_BN_Leaky(filters=num_filters * 2, kernel_size=(3, 3),
                                            block_id_str=block_id_str + '_2'),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    return x


def get_efficientnet_backbone_info(input_tensor):
    efficientnet = EfficientNetB1(input_tensor=input_tensor, weights='imagenet', include_top=False)

    f1_name = 'top_activation'
    f1_channel_num = 1280
    f2_name = 'block6a_expand_activation'
    f2_channel_num = 672
    f3_name = 'block4a_expand_activation'
    f3_channel_num = 240

    feature_map_info = {'f1_name': f1_name,
                        'f1_channel_num': f1_channel_num,
                        'f2_name': f2_name,
                        'f2_channel_num': f2_channel_num,
                        'f3_name': f3_name,
                        'f3_channel_num': f3_channel_num,
                        }

    return efficientnet, feature_map_info


def yolo4_predictions(feature_maps, feature_channel_nums, num_anchors, num_classes):
    f1, f2, f3 = feature_maps
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    x1 = make_yolo_spp_head(f1, f1_channel_num // 2)

    x1_upsample = compose(
        DarknetConv2D_BN_Leaky(f2_channel_num // 2, (1, 1)),
        UpSampling2D(2))(x1)

    x2 = DarknetConv2D_BN_Leaky(f2_channel_num // 2, (1, 1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    x2 = make_yolo_head(x2, f2_channel_num // 2)

    x2_upsample = compose(
        DarknetConv2D_BN_Leaky(f3_channel_num // 2, (1, 1)),
        UpSampling2D(2))(x2)

    x3 = DarknetConv2D_BN_Leaky(f3_channel_num // 2, (1, 1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    x3 = make_yolo_head(x3, f3_channel_num // 2)
    y3 = compose(
        DarknetConv2D_BN_Leaky(f3_channel_num, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv_3'))(x3)

    x3_downsample = compose(
        ZeroPadding2D(((1, 0), (1, 0))),
        DarknetConv2D_BN_Leaky(f2_channel_num // 2, (3, 3), strides=(2, 2)))(x3)

    x2 = Concatenate()([x3_downsample, x2])

    x2 = make_yolo_head(x2, f2_channel_num // 2)
    y2 = compose(
        DarknetConv2D_BN_Leaky(f2_channel_num, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv_2'))(x2)

    x2_downsample = compose(
        ZeroPadding2D(((1, 0), (1, 0))),
        DarknetConv2D_BN_Leaky(f1_channel_num // 2, (3, 3), strides=(2, 2)))(x2)

    x1 = Concatenate()([x2_downsample, x1])

    x1 = make_yolo_head(x1, f1_channel_num // 2)
    y1 = compose(
        DarknetConv2D_BN_Leaky(f1_channel_num, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1), name='predict_conv_1'))(x1)

    return y1, y2, y3


def create_model(inputs, model_name):
    inputs = Input(shape=inputs, name='image_input')
    net, feature_map_info = load_custom_model(inputs, model_name)
    print('backbone layers number: {}'.format(len(net.layers)))
    f1 = net.get_layer(feature_map_info['f1_name']).output
    f1_channel_num = feature_map_info['f1_channel_num']

    f2 = net.get_layer(feature_map_info['f2_name']).output
    f2_channel_num = feature_map_info['f2_channel_num']

    f3 = net.get_layer(feature_map_info['f3_name']).output
    f3_channel_num = feature_map_info['f3_channel_num']

    y1, y2, y3 = yolo4_predictions((f1, f2, f3), (f1_channel_num, f2_channel_num, f3_channel_num), 3, 1)

    return Model(inputs, [y1, y2, y3])
