from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os

app = Flask(__name__)

# Path to the trained model. Update this path if your model is stored elsewhere.
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'brain_tumor_model.h5')

# Load the trained model when the application starts.
# If the model file is missing, Flask will raise an error on startup.
model = load_model(MODEL_PATH)

LABELS = ['no', 'yes']


def prepare_image(file_stream):
    """Read and preprocess the uploaded image."""
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        image = prepare_image(file.stream)
        preds = model.predict(image)
        label = LABELS[int(np.argmax(preds))]
        confidence = float(np.max(preds))
        return render_template('result.html', label=label, confidence=confidence)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
