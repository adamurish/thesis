import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import csv


def load_imgs(img_dir):
    """
    Loads images with their labels and bounding boxes

    :param img_dir: the directory to search for .csv with image filenames
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
                    imgs.append(cv2.imread(os.path.join(root_dir, row['Filename'])))
                    labels.append(row['Annotation tag'])
                    bounding_boxes.append(((int(row['Upper left corner X']), int(row['Upper left corner Y'])),
                                           (int(row['Lower right corner X']), int(row['Lower right corner Y']))))
    return imgs, labels, bounding_boxes


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
        cv2.rectangle(img, bounding_boxes[i][0], bounding_boxes[i][1], (255, 0, 0), 2)
        plt.imshow(img)
        plt.xlabel(labels[i])
    plt.show()


imgs, labels, bounding_boxes = load_imgs('F:/signDatabasePublicFramesOnly/vid0')
plot_sample(labels, imgs, bounding_boxes, num=4, rows=2, cols=2)
