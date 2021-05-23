import os
from datetime import datetime

date_time_for_save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

BATCH_SIZE = 2
NUMBER_CLASSES = 2
INPUT_SHAPE_IMAGE = (256, 256, 3)
PROPORTION_TEST_IMAGES = 0.15
EPOCHS = 150
AUGMENTATION_DATA = False

DATA_PATH = 'data'
JSON_FILE_PATH = os.path.join(DATA_PATH, 'data.json')
IMAGES_PATH = os.path.join(DATA_PATH, 'images')
MASKS_PATH = os.path.join(DATA_PATH, 'masks')

LEARNING_RATE = 0.0001
BACKBONE = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax'
NAME_MODEL = 'Unet'
FULL_NAME_MODEL = NAME_MODEL + BACKBONE

MODELS_DATA = 'models_data'
TENSORBOARD_LOGS = os.path.join(MODELS_DATA, 'tensorboard_logs')
SAVE_MODELS = os.path.join(MODELS_DATA, 'save_models')
LOGS = os.path.join(MODELS_DATA, 'logs')

SAVE_CURRENT_MODEL = os.path.join(SAVE_MODELS, NAME_MODEL + '_' + str(ENCODER_WEIGHTS) + '_' +
                                  date_time_for_save + '_' + str(AUGMENTATION_DATA))
SAVE_CURRENT_TENSORBOARD_LOGS = os.path.join(TENSORBOARD_LOGS, NAME_MODEL + '_' + str(ENCODER_WEIGHTS) + '_' +
                                             date_time_for_save + '_' + str(AUGMENTATION_DATA))
