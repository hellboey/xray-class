
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
from matplotlib.image import imread
import tensorflow as tf
from keras.models import model_from_json

app = Flask(__name__)

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("COVIDNN6_weights.h5")
model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])


def preprocess_input(array):
    if np.amax(array)>1:
        array=array/255.0
    return array

def load_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path,target_size=(256,256))
    img_tensor = tf.keras.preprocessing.image.img_to_array(img)
    img_tensor = cv2.cvtColor(img_tensor,cv2.COLOR_BGR2GRAY)
    img_tensor = img_tensor/255.0
    img_tensor = np.expand_dims(img_tensor,axis = 0)
    img_tensor = np.expand_dims(img_tensor,axis = -1)

    return img_tensor

def predict_label(img_path):
    img=load_image(img_path)
    k=model.predict(img)

    if(k[0][0] > 0.05):
        return "positive"
    else:
        return "negative"

@app.route("/", methods=['GET', 'POST'])

def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path ="static/"+img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	app.run(debug = True)









