# web imports
import base64
import numpy as np
import time
import eventlet
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
from flask import Flask, render_template, url_for, jsonify, request

# import gluon utilities
import mxnet as mx
import gluoncv as gcv
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()

# import utilities

# Constant to define model to use
SSD_ID = 0
FASTER_RCNN_ID = 1
YOLOV3_ID = 2


# create SSD 512 pre-trained model
ssd = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
ssd.hybridize()

# Faster-RCNN pre-trained model
frcnn = gcv.model_zoo.get_model(
    'faster_rcnn_resnet50_v1b_voc', pretrained=True)
frcnn.hybridize()

# YoloV3 pre-trained model
yolo = gcv.model_zoo.get_model('yolo3_mobilenet1.0_voc', pretrained=True)
yolo.hybridize()


def base64_to_image(uri):
    """"read base64 encoded uri to image"""
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def image_preprocessing(frame, model_id):
    """Invokes the image pre-processing depending on the model_id used (Faster RCNN, YoloV3 or SSD 512)"""
    # image pre-processing (depends on the Model ID)
    if(model_id == SSD_ID):
        return gcv.data.transforms.presets.ssd.transform_test(
            frame, short=512, max_size=700)
    elif(model_id == FASTER_RCNN_ID):
        return gcv.data.transforms.presets.rcnn.transform_test(frame)
    else:
        return gcv.data.transforms.presets.yolo.transform_test(frame, short=512)


def model_use(x, model_id):
    """Use the model on the x data based on the model_id specified in parameters"""
    if(model_id == SSD_ID):
        return ssd(x)
    elif(model_id == FASTER_RCNN_ID):
        return frcnn(x)
    else:
        return yolo(x)


def format_predictions(class_IDs, scores, bounding_boxes, classes, threshold=0.5):
    """Formats the predictions of the gluonCV pre-trained model to the adapted results to return the right answer"""
    results = []  # results to return

    for i in range(len(scores)):
        score = scores[i][0]

        # Break if score is not better than threshold
        if score < threshold:
            break

        class_id = int(class_IDs[i][0])
        bbox = bounding_boxes[i]
        name = classes[class_id]
        results.append({
            'xmin': int(bbox[0]),
            'ymin': int(bbox[1]),
            'xmax': int(bbox[2]),
            'ymax': int(bbox[3]),
            'class': name,
            'score': score
        })

    return results


def predict_url(uri, model_id=SSD_ID):
    """prediction base64 image"""
    # convert uri to image in mx.nd array
    frame = base64_to_image(uri)
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

    # pre-processing
    x, frame = image_preprocessing(frame, model_id)

    # Run frame through network
    class_IDs, scores, bounding_boxes = model_use(x, model_id)

    img = gcv.utils.viz.cv_plot_bbox(
        frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=ssd.classes)
    gcv.utils.viz.cv_plot_image(img)
    cv2.waitKey()

    class_IDs = class_IDs.asnumpy()
    scores = scores.asnumpy()
    bounding_boxes = bounding_boxes.asnumpy()

    return format_predictions(class_IDs[0], scores[0], bounding_boxes[0], ssd.classes)

