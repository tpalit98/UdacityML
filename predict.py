import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import tensorflow_hub as hub
# TODO: Make all other necessary imports.
import numpy as np
import matplotlib.pyplot as plt
import time

import json

from PIL import Image

#from workspace_utils import keep_awake

import os, random

import warnings


warnings.filterwarnings("ignore")

#process images
def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    
    return image


#predict appropriate number of classes for the image using a pre-trained model
def predict(image_path, model, top_k, class_names):
    img = Image.open(image_path)
    test_image = np.asarray(img)
    processed_test_image = process_image(test_image)
    
    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, top_k)
    top_classes = [class_names[str(value)] for value in top_indices.cpu().numpy()[0]]
    top_values_final = top_values.numpy()[0] 
    
    return top_values_final, top_classes

#parse arguments
print("Please scroll to bottom for results")

parser = argparse.ArgumentParser()
parser.add_argument('image', action='store', default= './test_images/orchid.jpg', help='path to target image')
parser.add_argument('model', action='store',default= 'my_model.h5',  help='path to stored model')
parser.add_argument('--top_k', action='store', type=int, default=5, help='number of top classes')
parser.add_argument('--category_names', action='store',default='label_map.json', help='path to JSON file mapping labels to classes')

#assign variables
parse = parser.parse_args()
image_path = parse.image
model_path = parse.model
num_outputs = parse.top_k
cat_names = parse.category_names

#load model and process image
my_model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer': hub.KerasLayer})
im = Image.open(image_path)
test_image = np.asarray(im)
processed_test_image = process_image(test_image)

with open(cat_names, 'r') as f:
    class_names = json.load(f)

probs, classes = predict(image_path, my_model, num_outputs, class_names)


#results
print()
print("Results below!! Ignore text above!")
print("These are the top {} classes: {}".format(num_outputs, classes))
print("They have the following respective probabilities: {}.".format(probs))







