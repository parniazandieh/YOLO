# Import tensorflow
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.saving import register_keras_serializable

# Imports for visualizing predictions
import numpy as np
import pydicom
from skimage.transform import resize
from PIL import Image, ImageDraw

# Helper imports
import os
import pandas as pd
import time

# Path variables
shape_path = "trained_model/model_shape.json"
weights_path = "trained_model/model.weights.h5"
CSV_PATH = "label_data/CCC_clean.csv"
IMAGE_BASE_PATH = "../data/"

# Global variables
img_dims = 512

lambda_coord = 5
epsilon = 0.00001

# Define and register your custom loss function
@register_keras_serializable()
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


@register_keras_serializable()
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


@register_keras_serializable()
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


def load_model(shape_file, weights_file):
    """
    Load a tensorflow/keras model from an HDF5 file found at provided path.

    @param shape_file - path to JSON file of model architecture
    @param weights_file - path to HDF5 file of model weights
    @return model - a fully trained tf/keras model
    """
    print("Loading model from disk...")
    # load json and create model
    with open(shape_file, "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(
        loaded_model_json, custom_objects={
            "yolo_loss": YOLO_loss,
            "log_loss": log_loss,
            "iou_metric": IOU_metric,
        }
    )
    # Manually load weights into new model without checking the extension
    loaded_model.load_weights(weights_file)
    print("Complete!")
    return loaded_model


def is_dicom(im_path):
    """
    Check if the image specified by the path is a DICOM image.

    @param im_path - string path to an image file
    @return - boolean, True if is a dicom image, else False
    """
    # get file extension
    _, ext = os.path.splitext(im_path)

    # if file extension is empty or is .dcm, assume DICOM file
    # (we assume empty extension means DICOM because the format
    # of the DICOM data we are using was saved with an empty file
    # extension)
    return ext == ".dcm" or ext == ""


def load_image(im_path):
    """
    Load an image from provided path. Loads both DICOM and more
    common image formats into a numpy array.

    @param im_path - string path to the image file
    @return im - the image loaded as a numpy ndarray
    """
    if is_dicom(im_path):
        # load with pydicom
        im = pydicom.dcmread(im_path).pixel_array
        return im
    else:
        # load with Pillow
        im = Image.open(im_path)
        return np.array(im)


def pre_process(img):
    """
    Takes an image and preprocesses it to fit in the model

    @param img - a numpy ndarray representing an image
    @return - a shaped and normalized, grayscale version of img
    """
    # resize image to 512x512
    im_adjusted = resize(
        img, (img_dims, img_dims), anti_aliasing=True, preserve_range=True
    )

    # ensure image is grayscale (only has 1 channel)
    im_adjusted = im_adjusted.astype(np.float32)
    if len(im_adjusted.shape) >= 3:
        # squash 3 channel image to grayscale
        im_adjusted = np.dot(im_adjusted[..., :3], [0.299, 0.587, 0.114])

    # normalize the image to a 0-1 range
    if not np.amax(im_adjusted) < 1:  # check that image isn't already normalized
        if np.amin(im_adjusted) < 0:
            im_adjusted += np.amin(im_adjusted)
        im_adjusted /= np.amax(im_adjusted)

    # model requires 4D input; shape it to expected dims
    im_adjusted = np.reshape(im_adjusted, (1, img_dims, img_dims, 1))
    return im_adjusted


def normalize_image(img):
    """
    Normalize an image to the range of 0-255. This may help reduce the white
    washing that occurs with displaying DICOM images with PIL.

    @param img - a numpy array representing an image
    @return normalized - a numpy array whose elements are all within the range
                         of 0-255
    """
    # adjust for negative values
    normalized = img + np.abs(np.amin(img))

    # normalize to 0-1
    normalized = normalized.astype(np.float32)
    normalized /= np.amax(normalized)

    # stretch scale of range to 255
    normalized *= 255
    return normalized


def main():
    """
    Loads a saved Keras model from the trained_model/ directory and loads the
    image and ground truth points from the loaded dataset. It uses the
    model to make a bounding box prediction on the input image, and displays the
    image with the predicted and true bounding boxes.
    """
    # load a pretrained model from HDF5 file
    model = load_model(shape_path, weights_path)

    # load dataset for iterating paths
    data_frame = pd.read_csv(CSV_PATH)

    # iterate over all image paths
    for i in range(len(data_frame["imgPath"])):
        img = load_image(os.path.join(IMAGE_BASE_PATH, data_frame["imgPath"][i]))

        # ensure image fits model input dimensions
        preprocessed_img = pre_process(img)

        output = model.predict(preprocessed_img, batch_size=1)

        # un-normalize prediction to get plotable points
        points = np.array(output[0]) * img_dims
        points = list(points.astype(np.int32))

        # normalize image to prevent as much white-washing caused
        # by PIL lib as possible
        norm = normalize_image(img)

        # draw bbox of predicted points
        im = Image.fromarray(norm).convert("RGB")  # convert RGB for colored bboxes
        draw = ImageDraw.Draw(im)
        draw.rectangle(points, outline="#ff0000")  # red bbox

        # draw bbox of ground truth
        true_points = [
            int(data_frame["start_x"][i]),
            int(data_frame["start_y"][i]),
            int(data_frame["end_x"][i]) + int(data_frame["start_x"][i]),
            int(data_frame["end_y"][i]) + int(data_frame["start_y"][i]),
        ]

        print("True Points:", true_points)
        draw.rectangle(true_points, outline="#00ff00")  # green bbox

        im.show()
        im.save(f"./test-predict{i}.png")
        time.sleep(1)  # sleep to let user see/close image

        if i > 2:
            break 
        #only show 2 pics


"""
A program to use the trained and saved YOLO cancer detection model to make a 
bounding box prediction one image at time from the dataset specified by the 
data path global, iterating and showing ground truth bbox as well
"""
if __name__ == "__main__":
    main()
