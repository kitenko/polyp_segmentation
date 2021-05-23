import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models import build_model
from config import INPUT_SHAPE_IMAGE


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('--weights', type=str, default=None, help='Path for loading model weights.')
    parser.add_argument('--path_video', type=str, default=None, help='Path for loading video for test.')
    return parser.parse_args()


def preparing_frame(image: np.ndarray, model) -> np.ndarray:
    """
    This function prepares the image and makes a prediction.

    :param image: this is input image or frame.
    :param model: assembled model with loaded weights.
    :return: image with an overlay mask
    """

    image = cv2.resize(image, (INPUT_SHAPE_IMAGE[1], INPUT_SHAPE_IMAGE[0]))
    plt.figure(figsize=(20, 20))
    mask = model.predict(np.expand_dims(image, axis=0) / 255.0)[0]
    mask = np.where(mask >= 0.5, 1, 0)[:, :, 0]
    image[:, :, 2] = np.where(mask == 1, 200, image[:, :, 2])
    return image


def visualization() -> None:
    """
    This function captures video and resizes the image.
    """
    args = parse_args()
    model = build_model()
    model.load_weights(args.weights)

    cap = cv2.VideoCapture(args.path_video)

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            cv2.resize(frame, (INPUT_SHAPE_IMAGE[1], INPUT_SHAPE_IMAGE[0]))
            image = preparing_frame(image=frame, model=model)
            image = cv2.resize(image, (720, 720))
            cv2.imshow('frame', image)
            if cv2.waitKey(1) == ord('q'):
                break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
    visualization()
