from tensorflow.keras.applications.efficientnet import EfficientNetB1
from tensorflow.keras.applications.efficientnet import EfficientNetB5
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.densenet import DenseNet121


def load_EfficientNetB1(input_tensor):
    loaded_model = EfficientNetB1(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(640, 640, 3),
        pooling=None)

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
    return loaded_model, feature_map_info


def load_ResNet101(input_tensor):
    loaded_model = ResNet101(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(640, 640, 3),
        pooling=None)

    f1_name = 'conv5_block3_1_relu'
    f1_channel_num = 512
    f2_name = 'conv4_block22_2_relu'
    f2_channel_num = 256
    f3_name = 'conv3_block2_2_relu'
    f3_channel_num = 128

    feature_map_info = {'f1_name': f1_name,
                        'f1_channel_num': f1_channel_num,
                        'f2_name': f2_name,
                        'f2_channel_num': f2_channel_num,
                        'f3_name': f3_name,
                        'f3_channel_num': f3_channel_num,
                        }
    return loaded_model, feature_map_info


def load_ResNet50(input_tensor):
    loaded_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(640, 640, 3),
        pooling=None)

    f1_name = 'conv5_block1_1_relu'
    f1_channel_num = 512
    f2_name = 'conv4_block6_1_relu'
    f2_channel_num = 256
    f3_name = 'conv3_block3_2_relu'
    f3_channel_num = 128

    feature_map_info = {'f1_name': f1_name,
                        'f1_channel_num': f1_channel_num,
                        'f2_name': f2_name,
                        'f2_channel_num': f2_channel_num,
                        'f3_name': f3_name,
                        'f3_channel_num': f3_channel_num,
                        }
    return loaded_model, feature_map_info


def load_DenseNet121(input_tensor):
    loaded_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(640, 640, 3),
        pooling=None)

    f1_name = 'conv5_block9_0_relu'
    f1_channel_num = 768
    f2_name = 'conv4_block11_0_relu'
    f2_channel_num = 576
    f3_name = 'conv3_block9_0_relu'
    f3_channel_num = 384

    feature_map_info = {'f1_name': f1_name,
                        'f1_channel_num': f1_channel_num,
                        'f2_name': f2_name,
                        'f2_channel_num': f2_channel_num,
                        'f3_name': f3_name,
                        'f3_channel_num': f3_channel_num,
                        }
    return loaded_model, feature_map_info


def load_EfficientNetB5(input_tensor):
    loaded_model = EfficientNetB5(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(640, 640, 3),
        pooling=None)

    f1_name = 'block7b_activation'
    f1_channel_num = 3072
    f2_name = 'block5f_activation'
    f2_channel_num = 1056
    f3_name = 'block3a_activation'
    f3_channel_num = 240

    feature_map_info = {'f1_name': f1_name,
                        'f1_channel_num': f1_channel_num,
                        'f2_name': f2_name,
                        'f2_channel_num': f2_channel_num,
                        'f3_name': f3_name,
                        'f3_channel_num': f3_channel_num,
                        }
    return loaded_model, feature_map_info


def load_custom_model(input_tensor, model_name):
    if model_name == 'EfficientNetB1':
        loaded_model, feature_map_info = load_EfficientNetB1(input_tensor)
    elif model_name == 'ResNet50':
        loaded_model, feature_map_info = load_ResNet50(input_tensor)
    elif model_name == 'ResNet101':
        loaded_model, feature_map_info = load_ResNet101(input_tensor)
    elif model_name == 'DenseNet121':
        loaded_model, feature_map_info = load_DenseNet121(input_tensor)
    elif model_name == 'EfficientNetB5':
        loaded_model, feature_map_info = load_EfficientNetB5(input_tensor)
    else:
        print('Enter valid model Name - [EfficientNetB1,ResNet50,ResNet101,DenseNet121,VGG19]')
        exit()

    return loaded_model, feature_map_info
