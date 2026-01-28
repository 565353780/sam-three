import os
from typing import List


def loadImageFileNames(image_folder_path: str) -> List[str]:
    image_filename_list = os.listdir(image_folder_path)

    valid_image_filename_list = []

    for image_filename in image_filename_list:
        if image_filename.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
            continue

        valid_image_filename_list.append(image_filename)

    valid_image_filename_list.sort()

    return valid_image_filename_list
