import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
import cv2
import base64

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# Web imports
from flask import Flask, render_template, url_for, jsonify
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit
import eventlet

# parameters
img_height = 300
img_width = 300
confidence_threshold = 0.5

def readb64(uri):
    """ Converts the base64 encoded image in parameter to a cv2 image"""
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def build_ssd_model():
    # 1: Build the Keras model
    K.clear_session()  # Clear previous models from memory.

    model = ssd_300(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)

    # 2: Load the trained weights into the model.
    # TODO
    weights_path = 'VGG_VOC0712_SSD_300x300_iter_120000.h5'
    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    print('Loading model SSD DONE')
    return model


def predict(uri, model):
    # Convert base64 uri to image (cv2)
    original_img = readb64(uri)
    # resize to fit in SSD300
    resized_img = cv2.resize(original_img, (img_width, img_height))
    
    global graph
    with graph.as_default():
        # Prediction
        y_pred = model.predict(np.array([np.float32(resized_img)]))

        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold]
                     for k in range(y_pred.shape[0])]

    classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
    result = []
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * original_img.shape[1] / img_width
        ymin = box[3] * original_img.shape[0] / img_height
        xmax = box[4] * original_img.shape[1] / img_width
        ymax = box[5] * original_img.shape[0] / img_height
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        result.append({
            'xmin': xmin.item(),
            'ymin': ymin.item(),
            'xmax': xmax.item(),
            'ymax': ymax.item(),
            'class': classes[int(box[0])],
            'score': box[1].item()
        })
    return result


# building SSD300
model = build_ssd_model()

graph = tf.get_default_graph() # to make it work with Flask

# boxes = predict('data:image/png;base64,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', model)
# print(boxes)
# Web & websocket config
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins = '*')

# detection
@socketio.on('detection')
def detection(data):
    boxes = predict(data, model)
    emit('detected', boxes)


if __name__ == '__main__':
    socketio.run(app, debug=True)
