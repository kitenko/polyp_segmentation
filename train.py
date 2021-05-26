import os
import argparse
from typing import Tuple

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping

from config import (JSON_FILE_PATH, EPOCHS, SAVE_CURRENT_MODEL, TENSORBOARD_LOGS, LEARNING_RATE, MODELS_DATA,
                    SAVE_MODELS, SAVE_CURRENT_TENSORBOARD_LOGS, FULL_NAME_MODEL, INPUT_SHAPE_IMAGE, SAVE_CURRENT_LOGS)
from models import build_model
from data_generator import DataGenerator
from logcallback import LogCallback


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('--train', default=False, action='store_true',
                        help='if you use this flag, then the training will be performed automatically with input shape'
                             '(256*256*3))')
    parser.add_argument('--train_dif_shape', default=False, action='store_true',
                        help='if you use this flag, then the training will be performed automatically with a different '
                             'input shape from (256*256*3) before (512*512*3)')
    return parser.parse_args()


def train(dataset_path_json: str, input_shape_image: Tuple[int, int, int] = INPUT_SHAPE_IMAGE) -> None:
    """
    Training to classify generated images.

    :param dataset_path_json: path to json file.
    :param input_shape_image: this is image shape (height, width, channels).
    """
    # create dirs
    for p in [TENSORBOARD_LOGS, MODELS_DATA, SAVE_MODELS, SAVE_CURRENT_MODEL, SAVE_CURRENT_TENSORBOARD_LOGS,
              SAVE_CURRENT_LOGS + '_' + str(INPUT_SHAPE_IMAGE)]:
        os.makedirs(p, exist_ok=True)

    train_data_gen = DataGenerator(json_path=dataset_path_json, is_train=True, image_shape=input_shape_image)
    test_data_gen = DataGenerator(json_path=dataset_path_json, is_train=False, image_shape=input_shape_image)

    model = build_model(image_shape=input_shape_image)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=sm.losses.categorical_focal_jaccard_loss,
                  metrics=['accuracy', sm.metrics.iou_score, sm.metrics.precision, sm.metrics.recall,
                           sm.metrics.f1_score]
                  )
    model.summary()
    early = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='auto')
    checkpoint_filepath = os.path.join(SAVE_CURRENT_MODEL, FULL_NAME_MODEL + '_' + str(input_shape_image) + '.h5')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='iou_score',
        mode='max',
        save_best_only=True
    )

    tensor_board = tf.keras.callbacks.TensorBoard(SAVE_CURRENT_TENSORBOARD_LOGS + '_' +
                                                  str(input_shape_image), update_freq='batch')
    with LogCallback(logs_save_path=SAVE_CURRENT_LOGS + '_' + str(INPUT_SHAPE_IMAGE)) as call_back:
        model.fit_generator(generator=train_data_gen, validation_data=test_data_gen, validation_freq=1,
                            validation_steps=len(test_data_gen), epochs=EPOCHS, workers=8,
                            callbacks=[early, model_checkpoint_callback, tensor_board, call_back])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    if parse_args().train is True:
        train(JSON_FILE_PATH, input_shape_image=INPUT_SHAPE_IMAGE)

    if parse_args().train_dif_shape is True:
        for i in range(5):
            input_shape = [(INPUT_SHAPE_IMAGE[0] + (64 * i), INPUT_SHAPE_IMAGE[1] +
                            (64 * i), 3) for i in range(0, 5, 1)]
            train(dataset_path_json=JSON_FILE_PATH, input_shape_image=input_shape[i])
