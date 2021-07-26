import numpy as np
from tensorflow.keras.callbacks import Callback
class DatasetShuffleCallBack(Callback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        np.random.shuffle(self.dataset)


    

