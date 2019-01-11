import tensorflow as tf
from tensorflow import keras
import skimage.transform
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2


# taken from https://github.com/waleedka/traffic-signs-tensorflow/blob/master/notebook1.ipynb
def load_data(data_dir):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


# taken from https://github.com/waleedka/traffic-signs-tensorflow/blob/master/notebook1.ipynb
def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()


img = cv2.imread('F:/BelgianSigns/Training/00000/01153_00000.ppm')
print(img)
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
train_images, train_labels = load_data('F:/BelgianSigns/Training')
print(np.asarray(train_images).shape)
# test_images, test_labels = load_data('F:/BelgianSigns/Testing')
# display_images_and_labels(train_images, train_labels)
# resized_train_images = [skimage.transform.resize(image, (32, 32), mode='constant')
#                         for image in train_images]
# print(resized_train_images.shape())
# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
# model.add(keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(keras.layers.Dense(62, activation=tf.nn.softmax))
# model.compile(optimizer=tf.train.AdamOptimizer(),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(resized_train_images, train_labels, epochs=5)
