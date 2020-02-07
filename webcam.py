import argparse
import time
import mxnet as mx
import gluoncv as gcv
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()


def get_arguments():
    # parser definition
    parser = argparse.ArgumentParser(description='Creates a webcam detection.')
    parser.add_argument(
        '-n', '--network', help='The name of the network to use ("ssd", "frcnn" or "yolo")', default='yolo')
    args = parser.parse_args()

    if args.network != 'yolo' and args.network != 'frcnn' and args.network != 'ssd':
        return 'ssd'
    else:
        return args.network


def get_model(network):
    """choose the network depending on the 'network argumment'"""
    if network == 'yolo':
        yolo = gcv.model_zoo.get_model(
            'yolo3_mobilenet1.0_voc', pretrained=True, ctx=mx.gpu(0))
        yolo.hybridize()
        return yolo
    elif network == 'ssd':
        ssd = gcv.model_zoo.get_model(
            'ssd_512_mobilenet1.0_voc', pretrained=True, ctx=mx.gpu(0))
        ssd.hybridize()
        return ssd
    else:
        frcnn = gcv.model_zoo.get_model(
            'faster_rcnn_resnet50_v1b_voc', pretrained=True, ctx=mx.gpu(0))
        frcnn.hybridize()
        return frcnn


def pre_processing(net, name, frame):
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

    # Image pre-processing
    if name == 'yolo':
        rgb_nd, frame = gcv.data.transforms.presets.yolo.transform_test(
            frame, short=512)

    elif name == 'ssd':
        rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(
            frame, short=512, max_size=700)

    else:
        rgb_nd, frame = gcv.data.transforms.presets.rcnn.transform_test(frame)

    # Use the model to make the prediction
    rgb_nd = rgb_nd.as_in_context(mx.gpu(0))
    return rgb_nd, frame


network = get_arguments()
net = get_model(network)

# Launch video capture
# Load the webcam handler
cap = cv2.VideoCapture(0)
time.sleep(1)  # letting the camera autofocus

while(cap.isOpened()):
    ret, frame = cap.read()

    # break if error in reading camera
    if not ret:
        break
    
    # pre-processing
    rgb_nd, frame = pre_processing(net, network, frame)

    # make prediction
    class_IDs, scores, bounding_boxes = net(rgb_nd)

    # Display the result
    img = gcv.utils.viz.cv_plot_bbox(
        frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
    gcv.utils.viz.cv_plot_image(img)
    k = cv2.waitKey(1)

    # ESC key to stop webcam
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
