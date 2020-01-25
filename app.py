from flask import Flask, render_template, url_for, jsonify
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit
import eventlet

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video-detection')
def video_detection():
    return render_template('video-detection/video-detection.html')


@app.route('/image-detection')
def image_detection():
    return render_template('image-detection/image-detection.html')


@app.route('/webcam-detection')
def webcam_detection():
    return render_template('webcam-detection/webcam-detection.html')


@socketio.on('detection')
def detection(data):
    emit('detected', [
        {
            'xmin': 40,
            'ymin': 100,
            'xmax': 120,
            'ymax': 240,
            'class': 'Saloperie',
            'score': 99
        }
    ])


if __name__ == '__main__':
    socketio.run(app, debug = True)
