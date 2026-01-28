import os
import cv2
from tqdm import trange

from sam_three.Method.image import toMaskImage, toMaskedImage
from sam_three.Method.io import loadImageFileNames
from sam_three.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    model_file_path = home + '/chLi/Model/SAM3/sam3/sam3.pt'
    image_folder_path = home + "/chLi/Dataset/GS/haizei_1_v4/gs/images/"
    mask_folder_path = home + "/chLi/Dataset/GS/haizei_1_v4/gs/masks/"
    masked_image_folder_path = home + "/chLi/Dataset/GS/haizei_1_v4/gs/masked_images/"


    detector = Detector(model_file_path)

    valid_image_filename_list = loadImageFileNames(image_folder_path)

    valid_image_file_path_list = [
        image_folder_path + image_filename for image_filename in valid_image_filename_list
    ]

    masks = detector.detectImageFiles(valid_image_file_path_list)

    os.makedirs(mask_folder_path, exist_ok=True)
    print('start save mask...')
    for i in trange(len(valid_image_file_path_list)):
        cv2.imwrite(mask_folder_path + valid_image_filename_list[i], toMaskImage(masks[i]))

    os.makedirs(masked_image_folder_path, exist_ok=True)
    print('start save masked images...')
    for i in trange(len(valid_image_file_path_list)):
        # 读取原始图像
        image = cv2.imread(valid_image_file_path_list[i])
        mask = masks[i]

        masked_image = toMaskedImage(image, mask)

        cv2.imwrite(masked_image_folder_path + valid_image_filename_list[i], masked_image)
    return True
