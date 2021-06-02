import os
import time
import argparse
from tqdm import tqdm

import cv2
import numpy as np
import tensorflow as tf
import segmentation_models as sm

from config import INPUT_SHAPE_IMAGE, JSON_FILE_PATH
from src import build_model
from src import DataGenerator


def preparing_frame(image: np.ndarray, model) -> np.ndarray:
    """
    This function prepares the image and makes a prediction.

    :param image: this is input image or frame.
    :param model: assembled model with loaded weights.
    :return: image with an overlay mask
    """

    image = cv2.resize(image, (INPUT_SHAPE_IMAGE[1], INPUT_SHAPE_IMAGE[0]))
    mask = model.predict(np.expand_dims(image, axis=0) / 255.0)[0]
    mask = np.where(mask >= 0.5, 1, 0)[:, :, 0]
    image[:, :, 2] = np.where(mask == 1, 200, image[:, :, 2])
    return image


def visualization() -> None:
    """
    This function captures video and resizes the image.
    """
    model = build_model()
    model.load_weights(args.weights)

    cap = cv2.VideoCapture(args.path_video)

    if not cap.isOpened():
        print("Error opening video stream or file")

    prev_frame_time = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        cv2.resize(frame, (INPUT_SHAPE_IMAGE[1], INPUT_SHAPE_IMAGE[0]))
        image = preparing_frame(image=frame, model=model)
        image = cv2.resize(image, (720, 720))
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(image, str(int(fps)) + ':fps', (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def test_metrics_and_time(mode: str) -> None:
    """
    This function calculates the average value of loss and metrics as well as inference time and average fps.

    :param mode: depending on the mode ('metrics', 'time'), the function counts (loss, metrics) or time and average fps.
    """
    data_gen = DataGenerator(batch_size=1, json_path=os.path.join(args.data_path, JSON_FILE_PATH), is_train=False)
    model = build_model()
    model.load_weights(args.weights)
    model.compile(loss=tf.keras.losses.binary_crossentropy, metrics=[
                        'accuracy', sm.metrics.iou_score, sm.metrics.precision, sm.metrics.recall,  sm.metrics.f1_score
    ])

    if mode == 'metrics':
        print(model.evaluate(data_gen, workers=8))

    elif mode == 'time':
        all_times = []
        for i in tqdm(range(len(data_gen))):
            images, _ = data_gen[i]
            start_time = time.time()
            model.predict(images)
            finish_time = time.time()
            all_times.append(finish_time - start_time)
        all_times = all_times[5:]
        message = '\nMean inference time: {:.04f}. Mean FPS: {:.04f}.\n'.format(
            np.mean(all_times),
            len(all_times) / sum(all_times))
        print(message)


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('--weights', type=str, default=None, help='Path for loading model weights.')
    parser.add_argument('--path_video', type=str, default=None, help='Path for loading video for test.')
    parser.add_argument('--test_on_video', action='store_true', help='If the value is True, then the webcam will be '
                                                                     'used for the test.')
    parser.add_argument('--metrics', action='store_true', help='If the value is True, then the average '
                                                               'metrics on the validation dataset will be calculated.')
    parser.add_argument('--time', action='store_true', help='If the value is True, then the inference time and the '
                                                            'average fps on the validation dataset will be calculated.')
    parser.add_argument('--gpu', action='store_true', help='If True, then the gpu is used for the test.')
    parser.add_argument('--data_path', type=str, default='data', help='path to Dataset where there is a json file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.gpu is True:
        devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(devices[0], True)

    if args.test_on_video is True:
        visualization()
    if args.metrics is True:
        test_metrics_and_time('metrics')
    if args.time is True:
        test_metrics_and_time('time')
