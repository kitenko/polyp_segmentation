import os
import json
import random
import argparse

import cv2

from config import JSON_FILE_PATH, PROPORTION_TEST_IMAGES


def prepare_data(data_path: str, proportion_test_images: float = PROPORTION_TEST_IMAGES) -> None:
    """
    :param proportion_test_images: proportion of test images.
    :param data_path: path data.
    """

    path = []
    for root, dirs, _ in os.walk(data_path):
        for dir in dirs:
            if dir == 'images':
                path.append(os.path.join(root, dir))
            elif dir == 'masks':
                path.append(os.path.join(root, dir))

    # reading and shuffling files
    images = []
    for image_path in path:
        g = image_path.split('/')
        if 'images' in g:
            for mask_path in os.listdir(image_path):
                images.append(os.path.join(image_path, mask_path))

    shuffle_images = random.sample(images, len(images))

    # create dictionary
    train_test_json = {'train': [], 'test': []}

    # filling in dictionary for json file
    for j, image_path in enumerate(shuffle_images):
        try:
            mask_path = image_path.replace('images', 'masks')
            img_dict = {'image_path': image_path, 'mask_path': mask_path}
            if cv2.imread(image_path) is None:
                print('broken image: ' + image_path)
                continue
            elif cv2.imread(mask_path) is None:
                print('broken image: ' + mask_path)
                continue
            elif j < len(shuffle_images) * proportion_test_images:
                train_test_json['test'].append(img_dict)
            else:
                train_test_json['train'].append(img_dict)
        except KeyError:
            print(' no mask for ', image_path)

    # write json file
    with open(os.path.join(data_path, JSON_FILE_PATH), 'w') as f:
        json.dump(train_test_json, f, indent=4)


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser('script for model testing.')
    parser.add_argument('-p', '--data_path', type=str, default='data', help='path to Dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    prepare_data(data_path=args.data_path)
