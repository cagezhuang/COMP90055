
"""
 AUTHOR : Kage Zhuang
 PURPOSE : TensorFlow Lite model converter
"""

import os
import tensorflow as tf

# This section is for Mobilenet_v1
MODEL_DIR = os.getcwd() + '/mobilenet_1.0_224'
input_arrays = ["input"]
output_arrays = ["final_result"]

# This section is for Inception_v3
# MODEL_DIR = os.getcwd() + '/Inception_v3'
# input_arrays = ["Mul"]
# output_arrays = ["final_result"]

# Define input TF file name and output TF Lite file name
TF_MODEL_NAME = "retrained_graph.pb"
LITE_MODEL_NAME = 'converted_model.tflite'

# Create path for both TF and Lite files
TF_MODEL_PATH = os.path.join(MODEL_DIR, TF_MODEL_NAME)
LITE_MODE_PATH = os.path.join(MODEL_DIR, LITE_MODEL_NAME)

# Create converter and convert the model and final write the Lite model
converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(TF_MODEL_PATH, input_arrays, output_arrays)
tflite_model = converter.convert()
open(LITE_MODE_PATH, "wb").write(tflite_model)