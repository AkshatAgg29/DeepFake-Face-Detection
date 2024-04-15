import os
import cv2
import numpy as np
from flask import Flask, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__, static_url_path='/static')

UPLOAD_FOLDER = r'C:\Users\akagg\Desktop\rgar\data science  project\downloaded image'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the discriminator model
discriminator = load_model(r'C:\Users\akagg\discriminator.h5')

def test_discriminator(discriminator, images):
    images = (images - 0.5) * 2
    predictions = discriminator.predict(images)
    return predictions

SIZE = 128
def load_and_preprocess_image(image_path, size=(SIZE, SIZE)):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        img = (img - 127.5) / 127.5
        return img
    else:
        print(f"Warning: Unable to read image '{image_path}'")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Load and preprocess the uploaded image
            uploaded_image = load_and_preprocess_image(file_path)
            uploaded_image = np.expand_dims(uploaded_image, axis=0)  # Add batch dimension
            
            # Test the uploaded image using the discriminator
            prediction = test_discriminator(discriminator, uploaded_image)
            prediction_result = "Real" if prediction[0][0] > 0.8 else "Fake"
            
            # Redirect to result page with the prediction result and image path
            return redirect(url_for('show_result', prediction=prediction_result, image_path=file_path))

    return render_template('index.html')

@app.route('/result')
def show_result():
    prediction = request.args.get('prediction')
    image_path = request.args.get('image_path')
    return render_template('result.html', prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
