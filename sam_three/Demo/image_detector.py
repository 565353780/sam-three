import os
from tqdm import tqdm

from sam_three.Method.io import loadImageFileNames
from sam_three.Module.image_detector import ImageDetector


def demo():
    home = os.environ['HOME']
    model_file_path = home + '/chLi/Model/SAM3/sam3/sam3.pt'
    device = 'cuda:0'

    image_folder_path = home + "/chLi/Dataset/GS/haizei_1_v4/gs/images/"
    prompt = 'object'

    image_detector = ImageDetector(model_file_path, device)

    valid_image_filename_list = loadImageFileNames(image_folder_path)

    print('start detect images...')
    for valid_image_filename in tqdm(valid_image_filename_list):
        output = image_detector.detectImageFile(image_folder_path + valid_image_filename, prompt)

        if output is None:
            print('no instance found for image:', valid_image_filename)
            continue

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        print('found', len(masks), 'instances!')
    return True
