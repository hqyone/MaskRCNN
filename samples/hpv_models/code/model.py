if __name__ == '__main__':
    import matplotlib

    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
from skimage.measure import find_contours

import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

ROOT_DIR = os.path.abspath("../../")  # 指定根目录

# 导入Mask RCNN
sys.path.append(ROOT_DIR)  # 查找库的本地版本
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import time
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "../../mask_rcnn_hpv_0600.h5")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

RESULTS_DIR = os.path.join(ROOT_DIR, "results/hpv/")



def prepareFolders():
    pass


def train():
    pass

def predict():
    pass