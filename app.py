from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model 
import cv2
import numpy as np
import os

app = Flask(__name__)  

model = load_model('model_vgg16.h5')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image")
    else:
        print("Image shape:", image.shape)
    
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image_path, threshold=0.5):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    prediction_value = float(prediction[0][0])
    
    # Classify as pneumonia if prediction value is above threshold
    if prediction_value <= threshold:
        return 'Pneumonia'
    else:
        return 'Not Pneumonia'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    image_path = 'temp.jpg'  
    file.save(image_path)
    
    prediction = predict(image_path)
    
    os.remove(image_path)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True,template_folder='../templates')
