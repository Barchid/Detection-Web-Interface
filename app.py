from flask import Flask, render_template, url_for, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)


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


if __name__ == '__main__':
    app.run(debug = True)
