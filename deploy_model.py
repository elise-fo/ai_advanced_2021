import requests
import numpy as np
from tensorflow.image import resize_with_pad
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend 
from tensorflow.keras.models import load_model
from flask import Flask
from flask_ngrok import run_with_ngrok
from joblib import load
app = Flask(__name__)
run_with_ngrok(app)


def root_mean_squared_error(y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true)))
 
@app.route("/")
def home():
    return "API des chatons mignons"

@app.route('/predict', methods=['POST'])
def predict():
  chaton_mignon = request.json
	try:
	  image= np.array(chaton_mignon)
	except: 
		return -1
	
	size = image.shape

	if len(size)!=3:
		return -1
	
	if (size[0] > 4*size[1]) or (4*size[0] < size[1]):
		return -1
	
	if (size[0] < 128) or (size[1] < 128):
		return -1
	
	# chargement du modèle
	model = load_model('vgg16_model.h5', custom_objects = {'root_mean_squared_error':root_mean_squared_error})

	# prédiction
	image = resize_with_pad(image,128, 128)
	image = preprocess_input(image)
	image = np.expand_dims(image, axis=0)
  prediction = model.predict(image).tolist()[0][0]/100
  return prediction

app.run()

