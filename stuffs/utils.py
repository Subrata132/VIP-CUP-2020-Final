import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

def get_classes(classes_path):
    
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def get_dataset(annotation_file, shuffle):
    
    with open(annotation_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    if shuffle:
        seed=1234
        np.random.seed(seed)
        np.random.shuffle(lines)
        #np.random.seed(None)

    return lines


def get_optimizer(current_epoch):

    boundaries = [min(0,2-current_epoch)]
    values = [0.001,0.004]
    learning_rate= PiecewiseConstantDecay(boundaries=boundaries, values=values)

    optimizer = SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False)
    return optimizer

def add_metrics(model, metric_dict):
    '''
    add metric scalar tensor into model, which could be tracked in training
    log and tensorboard callback
    '''
    for (name, metric) in metric_dict.items():
        # seems add_metric() is newly added in tf.keras. So if you
        # want to customize metrics on raw keras model, just use
        # "metrics_names" and "metrics_tensors" as follow:
        #
        #model.metrics_names.append(name)
        #model.metrics_tensors.append(loss)
        model.add_metric(metric, name=name, aggregation='mean')



