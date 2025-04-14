import os
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# Load CSV files
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load trained model
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

# Prediction function
def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).view((-1, 3, 224, 224))
    output = model(input_data).detach().numpy()
    return np.argmax(output)

# Create Flask app
app = Flask(__name__)

# Prediction API route
@app.route('/predict', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = image.filename
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, filename)
    image.save(file_path)

    pred = prediction(file_path)

    return jsonify({
        'disease_name': disease_info['disease_name'][pred],
        'description': disease_info['description'][pred],
        'prevention': disease_info['Possible Steps'][pred],
        'disease_image_url': disease_info['image_url'][pred],
        'supplement_name': supplement_info['supplement name'][pred],
        'supplement_image_url': supplement_info['supplement image'][pred],
        'supplement_buy_link': supplement_info['buy link'][pred]
    })

# Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
