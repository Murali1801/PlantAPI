from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from .models import PredictionLog

# Create your views here.

MODEL_FOLDER = os.path.join(settings.BASE_DIR, 'model')
INPUT_SIZE = (64, 64)

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

# Use 224x224 for rice, 64x64 for others
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

class PredictView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        model_name = request.data.get('model', '').lower()
        image_file = request.data.get('image', None)
        if not model_name or not image_file:
            return Response({'error': 'Please provide both model and image parameters.'}, status=400)
        if model_name not in MODEL_FILES:
            return Response({'error': f"Model '{model_name}' not found. Choose from {list(MODEL_FILES.keys())}."}, status=400)
        model_path = os.path.join(MODEL_FOLDER, MODEL_FILES[model_name])
        if not os.path.exists(model_path):
            return Response({'error': f"Model file '{model_path}' not found."}, status=500)
        try:
            model = load_model(model_path)
            input_size = MODEL_INPUT_SIZES.get(model_name, (64, 64))
            img_array = load_and_preprocess_image_file(image_file, input_size)
            preds = model.predict(img_array)
            pred_class = int(np.argmax(preds, axis=1)[0])
            class_map = CLASS_MAP.get(model_name)
            if class_map:
                class_name = class_map.get(pred_class, f"Unknown ({pred_class})")
                # Log the prediction
                PredictionLog.objects.create(
                    model_name=model_name,
                    image_name=image_file.name,
                    predicted_class=class_name,
                    client_ip=request.META.get('REMOTE_ADDR')
                )
                return Response({'predicted_class': class_name})
            else:
                return Response({'predicted_class_index': pred_class})
        except Exception as e:
            return Response({'error': str(e)}, status=500)
