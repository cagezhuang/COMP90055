
"""
 AUTHOR : Kage Zhuang
 PURPOSE :  Create image augmentation for training images
            There are 5 methods used which are horizontal
            filp, vertical flip, rotate 90, random noise
            and greyscale.
"""

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# Define the training image folder
TRAINING_IMAGES_DIR = os.getcwd() + '/training_images'
count = 0

# Go through each folder in training image folder in case more than two classfications
for folder in next(os.walk(TRAINING_IMAGES_DIR))[1]:

    currentdirectory = TRAINING_IMAGES_DIR + "/" + folder
    directory = os.fsencode(currentdirectory)

    # Go through each image inside the folder
    for file in os.listdir(directory):

        filename = os.fsdecode(file)

        if filename.endswith(".jpg"):

            imgfile = Image.open(currentdirectory + "/" + filename)
            img = np.array(imgfile)

            # horizontal flip
            Image.fromarray(np.fliplr(img)).save(currentdirectory + "/" + filename.replace(".jpg", "_fliplr.jpg"))

            # vertical flip
            Image.fromarray(np.flipud(img)).save(currentdirectory + "/" + filename.replace(".jpg", "_flipud.jpg"))

            # rotate 90 degree
            Image.fromarray(np.rot90(img)).save(currentdirectory + "/" + filename.replace(".jpg", "_rot90.jpg"))

            # Add random noise
            noise = np.random.randint(20, size=(imgfile.height, imgfile.width, imgfile.layers), dtype='uint8')
            noiseimg = img + noise
            Image.fromarray(noiseimg).save(currentdirectory + "/" + filename.replace(".jpg", "_noise.jpg"))

            # Greyscale image using average method
            grayimg = img
            grayimg[:] = img.mean(axis=-1, keepdims=1)
            Image.fromarray(grayimg).save(currentdirectory + "/" + filename.replace(".jpg", "_gray.jpg"))


            # Print image number being processed. Just for fun.
            count += 1
            print(count)
        else:
            continue