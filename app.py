from flask import Flask, render_template, request
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image as img
from keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

app = Flask(__name__)

model = VGG16(weights='imagenet')


def predict_label(img_path):
    image = img.load_img(img_path, color_mode='rgb', target_size=(224, 224))
    x = img_to_array(image)
    x_expanded = np.expand_dims(x, axis=0)
    x_process = preprocess_input(x_expanded)
    features = model.predict(x_process)
    p = decode_predictions(features)
    return p[0][0]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)

        return render_template("index.html", prediction=p, img_path=img_path)
    else:
        return render_template("index.html", prediction='No Value of P', img_path='null')


if __name__ == '__main__':
    # app.debug = True
    app.run()
