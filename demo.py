import os
import cv2
from tqdm import trange

from sam_three.Module.detector import Detector

home = os.environ['HOME']
model_file_path = home + '/chLi/Model/SAM3/sam3/sam3.pt'
image_folder_path = home + "/chLi/Dataset/GS/haizei_1_v4/gs/images/"
mask_folder_path = home + "/chLi/Dataset/GS/haizei_1_v4/gs/masks/"
masked_image_folder_path = home + "/chLi/Dataset/GS/haizei_1_v4/gs/masked_images/"


detector = Detector(model_file_path)

image_filename_list = os.listdir(image_folder_path)

valid_image_filename_list = []

for image_filename in image_filename_list:
    if image_filename.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
        continue

    valid_image_filename_list.append(image_filename)

valid_image_filename_list.sort()

valid_image_file_path_list = [
    image_folder_path + image_filename for image_filename in valid_image_filename_list
]
valid_mask_file_path_list = [
    mask_folder_path + image_filename for image_filename in valid_image_filename_list
]

masks = detector.detectImageFiles(valid_image_file_path_list)

os.makedirs(mask_folder_path, exist_ok=True)
print('start save mask...')
for i in trange(len(valid_image_file_path_list)):
    mask = masks[i]
    # 将bool类型的mask转为0/255的uint8, 再转换为RGB二值图
    mask_uint8 = (mask.astype("uint8") * 255)
    mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(mask_folder_path + valid_image_filename_list[i], mask_rgb)

'''
IMG_WIDTH, IMG_HEIGHT = image_list[0].size

points_tensor = torch.tensor(
    [
        [0.5, 0.5],
    ],
    dtype=torch.float32,
)
# positive clicks have label 1, while negative clicks have label 0
points_labels_tensor = torch.tensor(
    [1],
    dtype=torch.int32,
)

response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=1,
        points=points_tensor,
        point_labels=points_labels_tensor,
        obj_id=0,
    )
)
out = response["outputs"]

# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)

# finally, we reformat the outputs for visualization and plot the outputs every 60 frames
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
'''
