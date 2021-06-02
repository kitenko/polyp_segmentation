import os
import json
import random
import argparse

import cv2

from config import JSON_FILE_PATH, IMAGES_PATH, MASKS_PATH, PROPORTION_TEST_IMAGES


def prepare_data(masks_path: str = MASKS_PATH, proportion_test_images: float = PROPORTION_TEST_IMAGES,
                 json_file_path: str = JSON_FILE_PATH, images_path: str = IMAGES_PATH) -> None:
    """

    :param masks_path: this is pass for masks files.
    :param proportion_test_images: proportion of test images.
    :param json_file_path: path to save json file.
    :param images_path: this is path for image files.
    """

    # reading and shuffling files
    count_images = os.listdir(images_path)
    shuffle_images = random.sample(count_images, len(count_images))

    # create dictionary
    train_test_json = {'train': [], 'test': []}

    # checking masks file in folder
    def find(name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(name)

    # filling in dictionary for json file
    for j, i in enumerate(shuffle_images):
        try:
            masks = find(i.rsplit(".", 1)[0] + '.jpg', masks_path)
            img_dict = {'image_path': os.path.join(images_path, i), 'mask_path': os.path.join(masks_path, masks)}
            if cv2.imread(os.path.join(images_path, i)) is None:
                print('broken image')
                continue
            elif j < len(shuffle_images) * proportion_test_images:
                train_test_json['test'].append(img_dict)
            else:
                train_test_json['train'].append(img_dict)
        except KeyError:
            print(' no mask for ', i)

    # write json file
    with open(json_file_path, 'w') as f:
        json.dump(train_test_json, f, indent=4)


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('--data_path', nargs='?', default=None, const='data', help='path to Dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    prepare_data(json_file_path=os.path.join(args.data_path, JSON_FILE_PATH),
                 images_path=os.path.join(args.data_path, IMAGES_PATH),
                 masks_path=os.path.join(args.data_path, MASKS_PATH))
