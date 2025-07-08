import os
import glob
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_FOLDER = "model"
TEST_FOLDER = "test"
INPUT_SIZE = (64, 64)

# Map model file to image prefix
MODEL_IMAGE_PREFIX = {
    "mazie.h5": "maize",
    "sunflower.h5": "sunflower",
    "wheat.h5": "wheat",
}

WHEAT_CLASSES = {
    0: "Brown rust",
    1: "Healthy",
    2: "Loose Smut",
    3: "Septoria",
    4: "Yellow rust",
}

SUNFLOWER_CLASSES = {
    0: "Healthy_sunflower",
    1: "rust",
    2: "sclerotinia",
}

MAIZE_CLASSES = {
    0: "Healthy",
    1: "downy_mildew",
    2: "maydis_leaf_blight",
    3: "rust",
    4: "turcicum_leaf_blight",
}

RICE_CLASSES = {
    0: "Bacterial Leaf Blight",
    1: "Brown Spot",
    2: "Healthy Rice Leaf",
    3: "Leaf Blast",
    4: "Leaf scald",
    5: "Narrow Brown Leaf Spot",
    6: "Rice Hispa",
    7: "Sheath Blight",
}


def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(INPUT_SIZE)
    img_array = np.array(img) / 255.0  # normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def main():
    for model_file, img_prefix in MODEL_IMAGE_PREFIX.items():
        model_path = os.path.join(MODEL_FOLDER, model_file)
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Skipping.")
            continue
        print(f"\nLoading model: {model_file}")
        model = load_model(model_path)
        print("Model input shape:", model.input_shape)
        pattern = os.path.join(TEST_FOLDER, f"{img_prefix}*.jpg")
        img_files = glob.glob(pattern)
        if not img_files:
            print(f"No test images found for prefix '{img_prefix}'.")
            continue
        for img_path in img_files:
            img_array = load_and_preprocess_image(img_path)
            preds = model.predict(img_array)
            pred_class = np.argmax(preds, axis=1)[0]
            if model_file == "wheat.h5":
                class_name = WHEAT_CLASSES.get(pred_class, f"Unknown ({pred_class})")
                print(
                    f"Image: {os.path.basename(img_path)} | Predicted class: {class_name}"
                )
            elif model_file == "sunflower.h5":
                class_name = SUNFLOWER_CLASSES.get(
                    pred_class, f"Unknown ({pred_class})"
                )
                print(
                    f"Image: {os.path.basename(img_path)} | Predicted class: {class_name}"
                )
            elif model_file == "mazie.h5":
                class_name = MAIZE_CLASSES.get(pred_class, f"Unknown ({pred_class})")
                print(
                    f"Image: {os.path.basename(img_path)} | Predicted class: {class_name}"
                )
            elif model_file == "rice.h5":
                class_name = RICE_CLASSES.get(pred_class, f"Unknown ({pred_class})")
                print(
                    f"Image: {os.path.basename(img_path)} | Predicted class: {class_name}"
                )
            else:
                print(
                    f"Image: {os.path.basename(img_path)} | Predicted class index: {pred_class}"
                )


if __name__ == "__main__":
    main()
