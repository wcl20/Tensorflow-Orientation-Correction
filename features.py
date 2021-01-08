import argparse
import os
import numpy as np
import random
import tqdm
from core.io import HDF5Writer
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True, help="Path to image dataset")
    parser.add_argument("--output", required=True, help="Path to output HDF5 file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch Size")
    args = parser.parse_args()

    # Load images
    img_paths = list(paths.list_images(args.img_dir))
    random.shuffle(img_paths)

    labels = [img_path.split(os.path.sep)[-2] for img_path in img_paths]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Dataset writer
    dims = (len(img_paths), 512 * 7 * 7)
    dataset = HDF5Writer(args.output, dims)
    dataset.store_class_names(label_encoder.classes_)

    # Feature extraction model
    model = VGG16(weights="imagenet", include_top=False)

    for i in tqdm.tqdm(np.arange(0, len(img_paths), args.batch_size)):

        # Image batch
        batch_img_paths = img_paths[i:i+args.batch_size]
        batch_labels = labels[i:i+args.batch_size]

        # Process batch
        batch_images = []
        for img_path in batch_img_paths:
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)
            batch_images.append(image)
        batch_images = np.vstack(batch_images)

        # Extract features
        batch_features = model.predict(batch_images, batch_size=args.batch_size)
        batch_features = batch_features.reshape((batch_features.shape[0], -1))

        dataset.add(batch_features, batch_labels)

    dataset.close()



if __name__ == '__main__':
    main()
