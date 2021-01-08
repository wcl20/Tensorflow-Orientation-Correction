# Tensorflow Orientation Correction

## Setup

Git clone repository
```bash
https://github.com/wcl20/Tensorflow-Orientation-Correction.git
```
Download Indoor Scene dataset [Link](http://web.mit.edu/torralba/www/indoor.html)
```
cd <Path to Tensorflow-Orientation-Correction>
mkdir dataset
mv <Indoor Scene dataset Images> dataset/images
```
Run nvidia docker container
```bash
docker run --gpus all -it --rm \
-v <Path to Tensorflow-Orientation-Correction>:/workspace \
nvcr.io/nvidia/tensorflow:20.11-tf2-py3
```
Install dependencies
```bash
apt-get update
apt-get install -y ffmpeg libsm6 libxext6
pip3 install --upgrade pip
pip3 install opencv-python pillow imutils tqdm h5py scikit-learn
```

## Build Project 
```bash
python3 build.py --img-dir dataset/images --output dataset/rotated_images
```

## Extract Features using VGG Model
```bash
python3 features.py --img-dir dataset/rotated_images --output dataset/features.hdf5
```

## Train model
```bash
python3 train.py --db dataset/features.hdf5 --output model.pickle
```

## Test model
```bash
python3 test.py --db dataset/features.hdf5 --img-dir dataset/rotated_images --model model.pickle
```
