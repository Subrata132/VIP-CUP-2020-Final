import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, \
    EarlyStopping, LambdaCallback
from model.EfficientNet import create_model
from stuffs.utils import get_optimizer, get_anchors, get_dataset, get_classes, add_metrics
from loss.loss import yolo3_loss

from data.data import data_generator_wrapper
from data.dataPrep import create_txt_file
from callbacks.callbacks import DatasetShuffleCallBack

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure GPU
# =====================================================================================#
tf.compat.v1.disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# =====================================================================================#

# Get attributes from cmd
# =====================================================================================#

parser = argparse.ArgumentParser()
parser.add_argument("--retrain", default=False, action="store_true")
parser.add_argument("--batch_size", type=int, required=False, default=8)
parser.add_argument("--epochs", type=int, required=False, default=200)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--weights", type=str, required=False, default=None)
parser.add_argument("--current_epoch", type=int, required=True,
                    help='epoch number of latest weights file (0 if training from the start)')
parser.add_argument("--train_file", type=str, required=False, default='train_data.txt')
parser.add_argument("--multigpu", default=False, action="store_true")
args = parser.parse_args()
retrain = args.retrain
batch_size = args.batch_size
epochs = args.epochs
model_name = args.model_name
weights = args.weights
current_epoch = args.current_epoch
train_file = args.train_file

# =====================================================================================#

# Load Dataset
# =====================================================================================#

try:
    file = open(train_file)
    file.close()

except:
    create_txt_file('train', True)

dataset = get_dataset(train_file, True)

num_train = int(0.8 * len(dataset))
num_val = len(dataset) - num_train
data_generator = data_generator_wrapper

# =====================================================================================#

# Create callbacks
# =====================================================================================#
log_dir = './logs/'

logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False,
                      update_freq='batch')
checkpoint = ModelCheckpoint(os.path.join(log_dir,
                                          'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-previous_epoch' + str(
                                              current_epoch) + '.h5'),
                             monitor='val_loss',
                             verbose=1,
                             save_weights_only=False,
                             save_best_only=True,
                             period=1)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, cooldown=0, min_lr=1e-10)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
callbacks = [logging, checkpoint, early_stopping]
shuffle_callback = DatasetShuffleCallBack(dataset)
callbacks.append(shuffle_callback)

# =====================================================================================#


# Set anchors & create model
# =====================================================================================#
anchors = [12, 16, 16, 27, 24, 14, 24, 37, 33, 21, 35, 58, 53, 40, 62, 77, 116, 120]
anchors = (np.asarray(anchors)).reshape((9, 2))
num_feature_layers = len(anchors) // 3
num_classes = 1
input_shape = (640, 640, 3)
image_shape = (640, 640)
label_smoothing = 0
elim_grid_sense = True
y_true = [Input(shape=(None, None, 3, num_classes + 5),
                name='y_true_{}'.format(l)) for l in range(num_feature_layers)]
if args.multigpu:
    devices_list = ["/gpu:{}".format(n) for n in range(args.gpu_num)]
    strategy = tf.distribute.MirroredStrategy(devices=devices_list)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = create_model(input_shape,model_name)
else:
    model = create_model(input_shape,model_name)

optimizer = get_optimizer(current_epoch)

model_loss, location_loss, confidence_loss, class_loss = Lambda(yolo3_loss, name='yolo_loss',
                                                                arguments={'anchors': anchors,
                                                                           'num_classes': num_classes,
                                                                           'ignore_thresh': 0.5,
                                                                           'label_smoothing': label_smoothing,
                                                                           'elim_grid_sense': elim_grid_sense})(
    [*model.output, *y_true])

model = Model([model.input, *y_true], model_loss)

loss_dict = {'location_loss': location_loss, 'confidence_loss': confidence_loss, 'class_loss': class_loss}
add_metrics(model, loss_dict)

model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

# =====================================================================================#


# Train & Test
# =====================================================================================#
if retrain == True:
    model.summary()
    if weights is None:
        model.fit(data_generator(dataset[:num_train], batch_size, image_shape, anchors, num_classes,
                                 multi_anchor_assign=False),
                  steps_per_epoch=max(1, num_train // batch_size),
                  validation_data=data_generator(dataset[num_train:], batch_size, image_shape, anchors, num_classes,
                                                 multi_anchor_assign=False),
                  validation_steps=max(1, num_val // batch_size),
                  epochs=epochs,
                  workers=1,
                  use_multiprocessing=False,
                  callbacks=callbacks,
                  max_queue_size=10)
    else:
        model.load_weights(weights, by_name=True)
        model.fit(data_generator(dataset[:num_train], batch_size, image_shape, anchors, num_classes,
                                 multi_anchor_assign=False),
                  steps_per_epoch=max(1, num_train // batch_size),
                  validation_data=data_generator(dataset[num_train:], batch_size, image_shape, anchors, num_classes,
                                                 multi_anchor_assign=False),
                  validation_steps=max(1, num_val // batch_size),
                  epochs=epochs,
                  workers=1,
                  use_multiprocessing=False,
                  callbacks=callbacks,
                  max_queue_size=10)

    # modelName=curdir+'/saved_models/'+'fishNet_retrained.h5'
    # model.save(modelName)


else:
    modelName = './saved_model/' + 'fishNet.h5'
    model = load_model(modelName)
    model.summary()

# =====================================================================================#
