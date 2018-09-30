import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import Flask
from flask import jsonify

app = Flask(__name__)

def get_model():
  global model
  model = load_model('../../softmax.h5')
  print("Model loaded")

def preprocess_image(image,target_size):
  if image.mode != "RGB":
      image = image.convert("RGB")
  image = image.resize(target_size)
  image = img_to_array(image)
  image = np.expand_dims(image,axis =0)

  return image

@app.route("/predict",methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image,target_size=(128,128))

    prediction = model.predict(processed_image)

    response = {
    'prediction':{
    'city':prediction[0][0],
    'bottles':prediction[0][1],
    'pots':prediction[0][2],
    'tyres':prediction[0][3]
    }
    }

    return jsonify(response)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    get_model()
    app.run(debug=True)
