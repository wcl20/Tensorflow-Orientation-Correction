import argparse
import cv2
import imutils
import numpy as np
import os
import random
import tqdm
from imutils import paths

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True, help="Path to image dataset")
    parser.add_argument("--output", required=True, help="Path to output images")
    args = parser.parse_args()

    img_paths = list(paths.list_images(args.img_dir))
    random.shuffle(img_paths)

    counter = {}

    for img_path in tqdm.tqdm(img_paths):
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue
        # Rotate image
        angle = np.random.choice([0, 90, 180, 270])
        image = imutils.rotate_bound(image, angle)

        # Save image
        output_dir = os.path.sep.join([args.output, str(angle)])
        os.makedirs(output_dir, exist_ok=True)
        filename, ext = os.path.splitext(img_path)
        output_path = os.path.sep.join([output_dir, f"image_{counter.get(angle, 0):05d}{ext}"])
        cv2.imwrite(output_path, image)

        count = counter.get(angle, 0)
        counter[angle] = count + 1








if __name__ == '__main__':
    main()
