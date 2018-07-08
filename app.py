from ObjectDetector import Detector
import io

from flask import Flask, render_template, request

from PIL import Image
from flask import send_file

app = Flask(__name__)

detector = Detector()


# detector.detectNumberPlate('twocar.jpg')


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def upload():
    if request.method == 'POST':
        file = Image.open(request.files['file'].stream)
        img = detector.detectObject(file)
        return send_file(io.BytesIO(img),attachment_filename='image.jpg',mimetype='image/jpg')


if __name__ == "__main__":
    app.run()
