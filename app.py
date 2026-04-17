from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing import image # pyright: ignore[reportMissingImports]
import numpy as np
import os
from PIL import Image
from datetime import datetime
import logging

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

try:
    modelvgg16 = load_model("modelvgg16.keras", compile=False)
    modelxception = load_model("modelxception.keras", compile=False)
    modelnasnet = load_model("modelnasnetmobile.keras", compile=False)
except Exception as e:
    print(f"Error loading models: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("cnn.html")

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("classifications.html")

@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'message': 'No image in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    name = request.form.get('name', 'Not provided')
    age = request.form.get('age', 'Not provided')
    bmi = request.form.get('bmi', 'Not provided')

    logging.info(f"Received data - Name: {name}, Age: {age}, BMI: {bmi}, File: {file.filename}")

    if file and allowed_file(file.filename):
        filename = "temp_image.png"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        img = Image.open(temp_path).convert('RGB')
        new_filename = datetime.now().strftime("%d%m%Y-%H%M%S") + ".png"
        predict_image_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        img.save(predict_image_path, format="png")
        img.close()
        
        class_names = ['NO', 'YES']
        
        def predict_with_model(model, path):
            input_shape = model.input_shape[1:3]
            img = image.load_img(path, target_size=input_shape)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 127.5 - 1
            return model.predict(x, verbose=0)
        
        prediction_array_nasnet = predict_with_model(modelnasnet, predict_image_path)
        prediction_array_vgg16 = predict_with_model(modelvgg16, predict_image_path)
        prediction_array_xception = predict_with_model(modelxception, predict_image_path)
        
        result = {
            'filename': url_for('static', filename='uploads/' + new_filename),
            'patient_info': {
                'name': name,
                'age': age,
                'bmi': bmi
            },
            'predictionnasnet': class_names[np.argmax(prediction_array_nasnet)],
            'confidencenasnet': '{:2.0f}%'.format(100 * np.max(prediction_array_nasnet)),
            'predictionvgg16': class_names[np.argmax(prediction_array_vgg16)],
            'confidencevgg16': '{:2.0f}%'.format(100 * np.max(prediction_array_vgg16)),
            'predictionxception': class_names[np.argmax(prediction_array_xception)],
            'confidencexception': '{:2.0f}%'.format(100 * np.max(prediction_array_xception)),
        }
        return jsonify({'data': result})
    
    return jsonify({'message': 'File type not allowed'}), 400

@app.errorhandler(404)
def not_found(error):
    logging.error(f"404 error: {request.path}")
    return "Not Found", 404

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000, debug=True)