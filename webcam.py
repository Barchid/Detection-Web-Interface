import mxnet as mx
import gluoncv as gcv
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()
import argparse

# parser definition
parser = argparse.ArgumentParser(description='Creates a webcam detection.')
parser.add_argument('--network', help='The name of the network to use ("ssd", "frcnn" or "yolo")')
args = parser.parse_args()
