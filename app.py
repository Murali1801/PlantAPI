from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
CORS(app, origins=["https://agri-scan-ai.vercel.app", "http://localhost", "http://localhost:3000", "http://127.0.0.1", "http://127.0.0.1:3000"])

MODEL_FOLDER = 'model'
MODEL_FILES = {
    'mazie': 'mazie.h5',
    'sunflower': 'sunflower.h5',
    'wheat': 'wheat.h5',
    'rice': 'rice.h5',
}

WHEAT_CLASSES = {
    0: 'Brown rust',
    1: 'Healthy',
    2: 'Loose Smut',
    3: 'Septoria',
    4: 'Yellow rust',
}
SUNFLOWER_CLASSES = {
    0: 'Healthy_sunflower',
    1: 'rust',
    2: 'sclerotinia',
}
MAIZE_CLASSES = {
    0: 'Healthy',
    1: 'downy_mildew',
    2: 'maydis_leaf_blight',
    3: 'rust',
    4: 'turcicum_leaf_blight',
}
RICE_CLASSES = {
    0: 'Bacterial Leaf Blight',
    1: 'Brown Spot',
    2: 'Healthy Rice Leaf',
    3: 'Leaf Blast',
    4: 'Leaf scald',
    5: 'Narrow Brown Leaf Spot',
    6: 'Rice Hispa',
    7: 'Sheath Blight',
}
CLASS_MAP = {
    'wheat': WHEAT_CLASSES,
    'sunflower': SUNFLOWER_CLASSES,
    'mazie': MAIZE_CLASSES,
    'rice': RICE_CLASSES,
}
MODEL_INPUT_SIZES = {
    'rice': (224, 224),
    'wheat': (64, 64),
    'sunflower': (64, 64),
    'mazie': (64, 64),
}

def load_and_preprocess_image_file(file, input_size):
    img = Image.open(file).convert('RGB')
    img = img.resize(input_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'model' not in request.form or 'image' not in request.files:
        return jsonify({'error': 'Please provide both model and image parameters.'}), 400
    model_name = request.form['model'].lower()
    image_file = request.files['image']
    if model_name not in MODEL_FILES:
        return jsonify({'error': f"Model '{model_name}' not found. Choose from {list(MODEL_FILES.keys())}."}), 400
    model_path = os.path.join(MODEL_FOLDER, MODEL_FILES[model_name])
    if not os.path.exists(model_path):
        return jsonify({'error': f"Model file '{model_path}' not found."}), 500
    try:
        model = load_model(model_path)
        input_size = MODEL_INPUT_SIZES.get(model_name, (64, 64))
        img_array = load_and_preprocess_image_file(image_file, input_size)
        preds = model.predict(img_array)
        pred_class = int(np.argmax(preds, axis=1)[0])
        class_map = CLASS_MAP.get(model_name)
        if class_map:
            class_name = class_map.get(pred_class, f"Unknown ({pred_class})")
            return jsonify({'predicted_class': class_name})
        else:
            return jsonify({'predicted_class_index': pred_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 