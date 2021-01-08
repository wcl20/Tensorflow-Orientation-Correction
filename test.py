import argparse
import cv2
import h5py
import imutils
import numpy as np
import os
import pickle
from imutils import paths
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to HDF5 file")
    parser.add_argument("--img-dir", required=True, help="Path to image dataset")
    parser.add_argument("--model", required=True, help="Path to model pickle")
    args = parser.parse_args()

    db = h5py.File(args.db)
    labels = [int(angle) for angle in db["class_names"][:]]
    db.close()

    print("[INFO] Sampling images ...")
    img_paths = list(paths.list_images(args.img_dir))
    img_paths = np.random.choice(img_paths, size=(10, ), replace=False)

    # Feature extraction model
    vgg = VGG16(weights="imagenet", include_top=False)
    model = pickle.loads(open(args.model, "rb").read())

    os.makedirs("output/original", exist_ok=True)
    os.makedirs("output/corrected", exist_ok=True)

    for i, img_path in enumerate(img_paths):

        image = cv2.imread(img_path)

        # Preprocess image
        input = load_img(img_path, target_size=(224, 224))
        input = img_to_array(input)
        input = np.expand_dims(input, axis=0)
        input = imagenet_utils.preprocess_input(input)
        # Extract features
        features = vgg.predict(input)
        features = features.reshape((features.shape[0], -1))
        # Determine rotation
        angle = model.predict(features)
        angle = labels[angle[0]]

        corrected_image = imutils.rotate_bound(image, 360 - angle)

        cv2.imwrite(f"output/original/image{i:05d}.jpg", image)
        cv2.imwrite(f"output/corrected/image{i:05d}.jpg", corrected_image)


if __name__ == '__main__':
    main()
