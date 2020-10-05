import tensorflow as tf  
import tensorflow.keras as keras 
import tensorflow.keras.applications as models
import os
import glob
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np
import json
import cv2
from core import debugger

keras_path = ""
tflite_path = ""
q_tflite_path = ""


keras_model = tf.keras.models.load_model(keras_path)
tflite_model = 