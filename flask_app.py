import eventlet
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
from flask import Flask, render_template, url_for, jsonify, request
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_data_generator import DataGenerator
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_loss_function.keras_ssd_loss import SSDLoss
from models.keras_ssd300 import ssd_300
import base64
import cv2
import numpy as np
from imageio import imread
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Classes of Pascal VOC
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

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


def predict(uri, model, img=None, use_img=False):
    # Convert base64 uri to image (cv2)
    original_img = img if use_img else readb64(uri)

    # resize to fit in SSD300
    resized_img = cv2.resize(original_img, (img_width, img_height))

    global graph
    with graph.as_default():
        # Prediction
        y_pred = model.predict(np.array([np.float32(resized_img)]))

        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold]
                         for k in range(y_pred.shape[0])]
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
graph = tf.get_default_graph()  # to make it work with Flask


def capture_detection(filename):
    """Captures the detection from the fresh saved video"""

    # define codecc & videoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    # read video
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()

        # Prevent from bug
        if ret == False:
            break

        # Create the videowriter (now we know the frame's size)
        if out == None:
            out = cv2.VideoWriter('output.avi', fourcc,
                                  20.0, (frame.shape[1], frame.shape[0]))

        boxes = predict('', model, img=frame, use_img=True)
        drawBoundingBoxes(boxes, frame)
        out.write(frame)
        print('Frame processed')

    # Release everything when job is finished
    cap.release()
    out.release()


# Web & websocket config
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')
CORS(app)


@app.route('/')
def index():
    return 'Index'

# detection
@socketio.on('detection')
def detection(data):
    boxes = predict(data, model)
    emit('detected', boxes)


@app.route('/video_detection', methods=['POST'])
def video_detection():
    # save uploaded video
    uploadedVideo = request.files['video']
    # save file as "upload.avi" (if 'avi' is the upload's file extension)
    filename = 'upload.' + uploadedVideo.filename.split('.')[-1]
    uploadedVideo.save(filename)

    # make detection
    capture_detection(filename)

    return jsonify({'file': filename})


def drawBoundingBoxes(boxes, frame):
    """Draw the bounding boxes in the specified frame in parameter"""
    for box in boxes:
        frame = cv2.rectangle(
            frame, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), (0, 255, 0), 3)
        txt = box['class'] + ' - ' + "%.2f" % box['score']
        posTxt = (int(box['xmin']), int(box['ymin']) + 10)
        cv2.putText(frame, txt, posTxt, cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == '__main__':
    socketio.run(app, debug=True)
