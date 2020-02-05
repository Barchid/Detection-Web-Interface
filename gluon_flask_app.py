# web imports
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

# create SSD 512 pre-trained model
ssd = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
ssd.hybridize()


def base64_to_image(uri):
    """"read base64 encoded uri to image"""
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def ssd_predict_uri(uri):
    """prediction base64 image"""
    # convert uri to image in mx.nd array
    frame = base64_to_image(uri)
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

    # image pre-processing for ssd
    rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(
        frame, short=512, max_size=700)

    # Run frame through network
    class_IDs, scores, bounding_boxes = ssd(rgb_nd)

    img = gcv.utils.viz.cv_plot_bbox(
        frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=ssd.classes)
    gcv.utils.viz.cv_plot_image(img)
    cv2.waitKey()

    print(class_IDS)
    print(scores)
    print(scores)
    print(bounding_boxes)
    print(ssd.classes)

    return img

to_test = "data:image/png;base64,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"
ssd_predict_uri(to_test)