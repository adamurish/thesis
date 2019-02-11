import os
import random

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import csv
import skimage.transform as st


def load_imgs(img_dir, label_filter='all', resize=False, resize_scale=0.5, img_size=None):
    """
    Loads images with their labels and bounding boxes

    :param img_dir: the directory to search for .csv with image filenames
    :param label_filter: specific label to pick
    :param resize: resize images?
    :param resize_scale: scale factor for resize
    :param img_size: size of images, only needed for resize (height, width)
    :return: tuple of images, labels, and bounding boxes
    """
    imgs = []
    labels = []
    bounding_boxes = []
    root_dir = os.path.join(img_dir, os.listdir(img_dir)[0])
    for f in os.listdir(root_dir):
        if f.endswith('.csv'):
            with open(os.path.join(root_dir, f), newline='') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=';')
                for row in reader:
                    if label_filter == 'all':
                        imgs.append(cv2.imread(os.path.join(root_dir, row['Filename'])) if not resize
                                    else st.resize(cv2.imread(os.path.join(root_dir, row['Filename'])),
                                                   (int(img_size[0]*resize_scale), int(img_size[1]*resize_scale))))
                        labels.append(row['Annotation tag'])
                        bounding_boxes.append(
                            scale_bounding_box((int(row['Upper left corner X']), int(row['Upper left corner Y'])),
                                               (int(row['Lower right corner X']), int(row['Lower right corner Y'])),
                                               resize, resize_scale))
                    else:
                        if str(row['Annotation tag']) == label_filter:
                            imgs.append(cv2.imread(os.path.join(root_dir, row['Filename'])) if not resize
                                        else st.resize(cv2.imread(os.path.join(root_dir, row['Filename'])),
                                                       (int(img_size[0] * resize_scale), int(img_size[1] * resize_scale))))
                            labels.append(row['Annotation tag'])
                            bounding_boxes.append(
                                scale_bounding_box((int(row['Upper left corner X']), int(row['Upper left corner Y'])),
                                                   (int(row['Lower right corner X']), int(row['Lower right corner Y'])),
                                                   resize, resize_scale))
    return np.array(imgs), np.array(labels), np.array(bounding_boxes)


def scale_bounding_box(upper_corner, lower_corner, scale, scale_factor):
    if not scale:
        return upper_corner, lower_corner
    new_up = (int(upper_corner[0] * scale_factor), int(upper_corner[1] * scale_factor))
    new_low = (int(lower_corner[0] * scale_factor), int(lower_corner[1] * scale_factor))
    return new_up, new_low


def read_csv(img_dir):
    """
    Reads rows from a .csv file

    :param img_dir: directory to look for the .csv file in
    :return: list of rows
    """
    root_dir = os.path.join(img_dir, os.listdir(img_dir)[0])
    rows = []
    for f in os.listdir(root_dir):
        if f.endswith('.csv'):
            with open(os.path.join(root_dir, f), newline='') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=';')
                for row in reader:
                    rows.append(row)
    return rows


def plot_sample(labels, imgs, bounding_boxes, num=25, rows=5, cols=5):
    """
    Plots an array of images with their respective labels

    :param bounding_boxes:
    :param labels: numpy array of labels
    :param imgs: numpy array of images
    :param num: amount to plot, default 25
    :param rows: amount of rows
    :param cols: amount of columns
    """
    plt.figure(figsize=(10, 10))
    for i in range(num):
        plt.subplot(rows, cols, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = cv2.merge((imgs[i][:, :, 2], imgs[i][:, :, 1], imgs[i][:, :, 0]))
        cv2.rectangle(img, (bounding_boxes[i][0], bounding_boxes[i][1]),
                      (bounding_boxes[i][2], bounding_boxes[i][3]), (1, 0, 0), 2)
        plt.imshow(img)
        plt.xlabel(labels[i])
    plt.show()


def plot_random_sample(imgs, num=25):
    """
    Plots an array of images with their respective labels

    :param imgs: numpy array of images
    :param num: amount to plot, default 25
    """
    s_indexes = random.sample(range(len(imgs)), num)
    s_images = np.array([imgs[i] for i in s_indexes])
    plt.figure(figsize=(10, 10))
    if num/5 > round(num/5):
        col = int(num/5) + 1
    else:
        col = round(num/5)
    for i in range(num):
        plt.subplot(5, col, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = cv2.merge((s_images[i][:, :, 2], s_images[i][:, :, 1], s_images[i][:, :, 0]))
        plt.imshow(img)
    plt.show()


imgs, labels, bounding_boxes =    load_imgs('F:/signDatabasePublicFramesOnly/vid0', label_filter='stop', resize=True, resize_scale=.25, img_size=(480, 704))
imgs2, labels2, bounding_boxes2 = load_imgs('F:/signDatabasePublicFramesOnly/vid1', label_filter='stop', resize=True, resize_scale=.25, img_size=(480, 704))
imgs3, labels3, bounding_boxes3 = load_imgs('F:/signDatabasePublicFramesOnly/vid2', label_filter='stop', resize=True, resize_scale=.25, img_size=(480, 704))
imgs4, labels4, bounding_boxes4 = load_imgs('F:/signDatabasePublicFramesOnly/vid3', label_filter='stop', resize=True, resize_scale=.25, img_size=(480, 704))
imgs5, labels5, bounding_boxes5 = load_imgs('F:/signDatabasePublicFramesOnly/vid4', label_filter='stop', resize=True, resize_scale=.25, img_size=(480, 704))
imgs6, labels6, bounding_boxes6 = load_imgs('F:/signDatabasePublicFramesOnly/vid5', label_filter='stop', resize=True, resize_scale=.25, img_size=(480, 704))
imgs = np.concatenate((imgs, imgs2, imgs3, imgs4, imgs5, imgs6), 0)
labels = np.concatenate((labels, labels2, labels3, labels4, labels5, labels6), 0)
bounding_boxes = np.concatenate((bounding_boxes, bounding_boxes2, bounding_boxes3, bounding_boxes4, bounding_boxes5, bounding_boxes6), 0)
bounding_boxes.resize((bounding_boxes.shape[0], 4))
test_imgs = imgs[-50:]
imgs = imgs[:-50]
test_labels = labels[-50:]
labels = labels[:-50]
test_bounding_boxes = bounding_boxes[-50:]
bounding_boxes = bounding_boxes[:-50]
print(imgs.shape)
print(labels.shape)
print(bounding_boxes[0])

# plot_sample(labels, imgs, bounding_boxes, num=25, rows=5, cols=5)
# plot_random_sample(imgs)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(12, (5, 5), input_shape=imgs[0].shape))
model.add(keras.layers.Activation(tf.nn.relu))
model.add(keras.layers.Conv2D(24, (4, 4)))
model.add(keras.layers.Activation(tf.nn.relu))
model.add(keras.layers.Conv2D(48, (3, 3)))
model.add(keras.layers.Activation(tf.nn.relu))
model.add(keras.layers.Conv2D(24, (3, 3)))
model.add(keras.layers.Activation(tf.nn.relu))
model.add(keras.layers.Conv2D(24, (3, 3)))
model.add(keras.layers.Activation(tf.nn.relu))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128))
model.add(keras.layers.Activation(tf.nn.relu))
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation(tf.nn.relu))
model.add(keras.layers.Dense(4))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(imgs, bounding_boxes, epochs=5)

plot_sample(test_labels, test_imgs, model.predict(test_imgs))
