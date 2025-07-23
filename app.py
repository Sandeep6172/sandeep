from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("models/image_classifier.h5")

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction[0])
    class_name = classes[class_index]

    return f"Predicted Object: {class_name}"

if __name__ == "__main__":
    app.run(debug=True)