import cv2
import os
import numpy as np
import skimage.transform
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random


def load_images(parent_dir, lim=9999, trim_size=(32, 32)):
    """
    Loads nested images in parent_dir using their directory name
    as their class name

    :param trim_size:
    :param parent_dir: top directory with class directories
    :param lim: amount of images to load per class
    :return: numpy array of images, numpy array of labels
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(parent_dir)
                   if os.path.isdir(os.path.join(parent_dir, d))]
    t_labels = []
    t_images = []

    for d in directories:
        count = 0
        label_dir = os.path.join(parent_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            if count < lim:
                t_labels.append(int(d))
                t_images.append(skimage.transform.resize(cv2.imread(f), trim_size, mode='constant'))
                count += 1
    return np.array(t_images), np.array(t_labels)


def plot_sample(labels, imgs, num=25):
    """
    Plots an array of images with their respective labels

    :param labels: numpy array of labels
    :param imgs: numpy array of images
    :param num: amount to plot, default 25
    """
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
        img = cv2.merge((imgs[i][:, :, 2], imgs[i][:, :, 1], imgs[i][:, :, 0]))
        plt.imshow(img)
        plt.xlabel(labels[i])
    plt.show()


def random_test(t_model: keras.Sequential, t_images, t_labels, k=10, ex_images=None):
    """
    Uses a keras Sequential model to perform tests on random
    images in a test image set

    :param ex_images:
    :param t_model: the keras model
    :param t_images: numpy array of test images
    :param t_labels: numpy array of test image labels
    :param k: amount to test
    """
    s_indexes = random.sample(range(len(t_images)), k)
    s_images = np.array([t_images[i] for i in s_indexes])
    s_labels = [t_labels[i] for i in s_indexes]
    p_labels = t_model.predict(s_images)
    f_labels = ['Actual: {0}, Predicted: {1}'.format(s_labels[i], np.argmax(p_labels[i])) for i in range(k)]
    plt.figure(figsize=(10, 10))
    if k/5 > round(k/5):
        col = int(k/5) + 1
    else:
        col = round(k/5)
    for i in range(k):
        plt.subplot(5, col, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if s_labels[i] == np.argmax(p_labels[i]):
            img = cv2.copyMakeBorder(s_images[i], 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 1, 0])
        else:
            img = s_images[i]
            if ex_images is not None:
                img = np.concatenate((img, ex_images[np.argmax(p_labels[i])]), axis=1)
            img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 1])
        img = cv2.merge((img[:, :, 2], img[:, :, 1], img[:, :, 0]))
        plt.imshow(img)
        plt.xlabel(f_labels[i])
    plt.show()


images, labels = load_images('F:/BelgianSigns/Training')

# plot_sample(labels, images)
# print(images.shape)
# print(labels.shape)

model = keras.Sequential()

model.add(keras.layers.Conv2D(12, (5, 5), input_shape=images[0].shape))
model.add(keras.layers.Activation(tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(24, (4, 4)))
model.add(keras.layers.Activation(tf.nn.relu))

model.add(keras.layers.Conv2D(48, (3, 3)))
model.add(keras.layers.Activation(tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(24, (3, 3)))
model.add(keras.layers.Activation(tf.nn.relu))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation(tf.nn.relu))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(62))
model.add(keras.layers.Activation(tf.nn.softmax))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_path = 'training/cp.ckpt'
if input('Retrain, or load weights? (r/l)') == 'r':
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
    model.fit(images, labels, epochs=30, callbacks=[cp_callback])
else:
    model.load_weights(checkpoint_path)

test_images, test_labels = load_images('F:/BelgianSigns/Testing')
example_images = load_images('F:/BelgianSigns/Testing', lim=1)[0]
loss, accuracy = model.evaluate(test_images, test_labels)
print('Loss: {0}, Accuracy: {1}%'.format(loss, int(accuracy * 100)))

random_test(model, test_images, test_labels, k=int(input('Amount?')), ex_images=example_images)
while input('Again? (y/n)') == 'y':
    random_test(model, test_images, test_labels, k=int(input('Amount?')), ex_images=example_images)
