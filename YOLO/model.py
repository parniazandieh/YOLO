import tensorflow as tf
import keras.backend as K
import math
import numpy as np
import pandas as pd
import pydicom
import os
import sys
import time
from sklearn.model_selection import train_test_split
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm


CSV_PATH = "label_data/CCC_clean.csv"
IMAGE_BASE_PATH = "../data/"
test_size_percent = 0.15  # percent of total data reserved for testing
mirror_im = False

# Loss
lambda_coord = 5
epsilon = 0.00001

# Learning
step_size = 1e-5
BATCH_SIZE = 5
num_epochs = 1

# Saving
shape_path = "trained_model/model_shape.json"
weight_path = "trained_model/model.weights.h5"

# TensorBoard
tb_graph = False
tb_update_freq = "batch"

print("Loading and processing data\n")
data_frame = pd.read_csv(CSV_PATH)
points = zip(
    data_frame["start_x"],
    data_frame["start_y"],
    data_frame["end_x"],
    data_frame["end_y"],
)
img_paths = data_frame["imgPath"]


def path_to_image(path):
    """
    Load a matrix of pixel values from the DICOM image stored at the
    input path.

    @param path - string, relative path (from IMAGE_BASE_PATH) to
                  a DICOM file
    @return image - numpy ndarray (int), 2D matrix of pixel
                    values of the image loaded from path
    """
    image = pydicom.dcmread(os.path.join(IMAGE_BASE_PATH, path)).pixel_array
    return image

def normalize_image(img):
    """
    Normalize the pixel values in img to be withing the range
    of 0 to 1.

    @param img - numpy ndarray, 2D matrix of pixel values
    @return img - numpy ndarray (float), 2D matrix of pixel values, every
                  element is valued between 0 and 1 (inclusive)
    """
    img = img.astype(np.float32)
    img += abs(np.amin(img))  # account for negatives
    img /= np.amax(img)
    return img

def normalize_points(points):
    """
    Normalize values in points to be within the range of 0 to 1.

    @param points - 1x4 tuple, elements valued in the range of 0
                    512 (inclusive). This is known from the nature
                    of the dataset used in this program
    @return - 1x4 numpy ndarray (float), elements valued in range
              0 to 1 (inclusive)
    """
    imDims = 512.0  # each image in our dataset is 512x512
    points = list(points)
    for i in range(len(points)):
        points[i] /= imDims
    return np.array(points).astype(np.float32)

"""
Convert the numpy array of paths to the DICOM images to pixel 
matrices that have been normalized to a 0-1 range.
Also normalize the bounding box labels to make it easier for
the model to predict on them.
"""
points = list(map(normalize_points, points))
imgs = list(map(path_to_image, img_paths))
imgs = list(map(normalize_image, imgs))

print("Number of images:", len(imgs))
print("Shape of each image:", imgs[0].shape if len(imgs) > 0 else "No images")

# Reshape input image data to 4D shape (as expected by the model)
imgs = np.array(imgs)
points = np.array(points)
imgs = imgs.reshape(-1, 512, 512, 1)

# Split the preprocessed data into train and test
train_imgs, test_imgs, train_points, test_points = train_test_split(
    imgs, points, test_size=test_size_percent, random_state=42
)

num_train_examples = len(train_imgs)
num_test_examples = len(test_imgs)

"""
Create generator for feeding the training data to the model
in batches. This specific generator is also capable of data
augmentation, which we do not use.
"""

generator = ImageDataGenerator(
    rotation_range=0,
    zoom_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    horizontal_flip=mirror_im,
    fill_mode="nearest",
)

print("Data preprocessing complete\n")

"""
Model definition according (approximately) to the YOLO model 
described by Redmon et al. in "You Only Look Once:
Unified, Real-Time Object Detection"
"""
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            64,
            (7, 7),
            padding="same",
            activation=tf.nn.relu,
            strides=2,
            input_shape=(512, 512, 1),
        ),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(192, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(128, (1, 1), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(256, (1, 1), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(256, (1, 1), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(256, (1, 1), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(256, (1, 1), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(256, (1, 1), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(512, (1, 1), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(1024, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(512, (1, 1), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(1024, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(512, (1, 1), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(1024, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(1024, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(
            1024, (3, 3), padding="same", strides=2, activation=tf.nn.relu
        ),
        tf.keras.layers.Conv2D(1024, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Conv2D(1024, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Flatten(),  # Flatten images into array for the fully connected layers
        tf.keras.layers.Dense(1024),
        tf.keras.layers.Dense(4096, activation=tf.keras.activations.linear),
        tf.keras.layers.Dense(
            4, activation=tf.nn.sigmoid
        ),  # 4 outputs: predict 4 points for a bounding box
    ]
)

"""
Our final layer predicts bounding box coordinates. We normalize 
the bounding box width and height by the image width and height
so that they fall between 0 and 1.

We use a sigmoid activation function for the final layer to 
facilitate learning of the normalized range of the output.

All the convolution layers use the rectified linear unit activation.
"""


# Custom Loss and Metric Functions
def log_loss(y_true, y_pred):
    """
    An implementation of the Unitbox negative log loss function proposed
    by Yu et al. in "Unitbox: an advanced object detection network".
    This loss function takes advantage of the observation that all the
    points in a bounding box prediction are highly correlated.

    @param y_true - BATCH_SIZEx4 Tensor object (float), the ground
                    truth labels for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
    @param y_pred - BATCH_SIZEx4 Tensor object (float), the model's
                    prediction for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
    @return loss - 1x1 Tensor object (float), valued between 0 and log(epsilon)
    """
    iou = IOU_metric(y_true, y_pred)
    iou = tf.where(tf.equal(tf.zeros_like(iou), iou), epsilon, iou)
    loss = -tf.math.log(iou)
    return loss


def YOLO_loss(y_true, y_pred):
    """
     An implementation of the Unitbox negative log loss function proposed
     by Yu et al. in "Unitbox: an advanced object detection network".
     This loss function takes advantage of the observation that all the
     points in a bounding box prediction are highly correlated.
     This loss function is very closely related to the mean squared error.

    @param y_true - BATCH_SIZEx4 Tensor object (float), the ground
                     truth labels for the bounding boxes around the
                     tumor in the corresponding images. Value order:
                     (x_top_left, y_top_left, x_bot_right, y_bot_right)
     @param y_pred - BATCH_SIZEx4 Tensor object (float), the model's
                     prediction for the bounding boxes around the
                     tumor in the corresponding images. Value order:
                     (x_top_left, y_top_left, x_bot_right, y_bot_right)
     @return loss - 1x1 Tensor object (float), value range 0-inf
    """
    x_LT = y_true[:, 0]
    y_UT = y_true[:, 1]
    x_RT = y_true[:, 2]
    y_LT = y_true[:, 3]

    x_LP = y_pred[:, 0]
    y_UP = y_pred[:, 1]
    x_RP = y_pred[:, 2]
    y_LP = y_pred[:, 3]

    x_Pmid = tf.math.add(x_LP, tf.math.divide(tf.math.subtract(x_RP, x_LP), 2))
    x_Tmid = tf.math.add(x_LT, tf.math.divide(tf.math.subtract(x_RT, x_LT), 2))
    y_Pmid = tf.math.add(y_UP, tf.math.divide(tf.math.subtract(y_LP, y_UP), 2))
    y_Tmid = tf.math.add(y_UT, tf.math.divide(tf.math.subtract(y_LT, y_UT), 2))

    x_mid_sqdiff = tf.math.square(tf.math.subtract(x_Pmid, x_Tmid))
    y_mid_sqdiff = tf.math.square(tf.math.subtract(y_Pmid, y_Tmid))

    first_term = tf.math.add(x_mid_sqdiff, y_mid_sqdiff)

    x_Pwidth = tf.math.sqrt(tf.math.abs(tf.math.subtract(x_RP, x_LP)))
    x_Twidth = tf.math.sqrt(tf.math.abs(tf.math.subtract(x_RT, x_LT)))
    y_Pheight = tf.math.sqrt(tf.math.abs(tf.math.subtract(y_UP, y_LP)))
    y_Theight = tf.math.sqrt(tf.math.abs(tf.math.subtract(y_UT, y_LT)))

    second_term = tf.math.add(
        tf.math.square(tf.math.subtract(x_Pwidth, x_Twidth)),
        tf.math.square(tf.math.subtract(y_Pheight, y_Theight)),
    )

    loss = tf.math.multiply(tf.math.add(first_term, second_term), lambda_coord)
    return loss


def IOU_metric(y_true, y_pred):
    """
    Compute the intersection over the union of the true and
    the predicted bounding boxes. Output in range 0-1;
    1 being the best match of bounding boxes (perfect alignment),
    0 being worst (no intersection at all).

    @param y_true - BATCH_SIZEx4 Tensor object (float), the ground
                    truth labels for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
    @param y_pred - BATCH_SIZEx4 Tensor object (float), the model's
                    prediction for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
    @return iou - 1x1 Tensor object (float), value being the mean
                  IOU for all image in the batch, and is within the
                  range of 0-1 (inclusive).
    """
    x_LT = y_true[:, 0]
    y_UT = y_true[:, 1]
    x_RT = y_true[:, 2]
    y_LT = y_true[:, 3]

    x_LP = y_pred[:, 0]
    y_UP = y_pred[:, 1]
    x_RP = y_pred[:, 2]
    y_LP = y_pred[:, 3]

    true_area = tf.math.multiply(
        tf.math.subtract(x_RT, x_LT), tf.math.subtract(y_LT, y_UT)
    )
    pred_area = tf.math.multiply(
        tf.math.subtract(x_RP, x_LP), tf.math.subtract(y_LP, y_UP)
    )

    x_all = tf.stack([x_LT, x_RT, x_LP, x_RP], axis=1)
    y_all = tf.stack([y_UT, y_LT, y_UP, y_LP], axis=1)

    x_max = tf.math.reduce_max(x_all, axis=1)
    x_min = tf.math.reduce_min(x_all, axis=1)
    y_max = tf.math.reduce_max(y_all, axis=1)
    y_min = tf.math.reduce_min(y_all, axis=1)

    intersect_width = tf.math.subtract(x_max, x_min)
    intersect_height = tf.math.subtract(y_max, y_min)
    intersect_area = tf.math.multiply(intersect_width, intersect_height)

    union_area = tf.math.subtract(tf.math.add(true_area, pred_area), intersect_area)

    return tf.math.divide(intersect_area, union_area)


print("Compiling model...\n")

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=step_size),
    loss=YOLO_loss,
    metrics=[log_loss, IOU_metric],
)

print("Model compilation complete\n")

log_dir = os.path.join("logs", "fit", time.strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, update_freq=tb_update_freq
)

### TRAIN THE MODEL ###
print("Training model...\n")

# Convert the data into a format suitable for the generator
train_data_generator = generator.flow(train_imgs, train_points, batch_size=BATCH_SIZE)

# Adjust the steps_per_epoch to match the new batch size
steps_per_epoch = math.ceil(num_train_examples / BATCH_SIZE)

history = model.fit(
    train_data_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    validation_data=(test_imgs, test_points),
    callbacks=[tensorboard_callback],
    verbose=1,
)

print("Model training complete\n")
print("Saving model...\n")

model_json = model.to_json()
with open(shape_path, "w") as json_file:
    json_file.write(model_json)
model.save_weights(weight_path)

print("Model saved to disk\n")
