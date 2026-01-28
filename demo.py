import os
import cv2
import torch
from PIL import Image
from tqdm import trange

from sam3.visualization_utils import prepare_masks_for_visualization
from sam3.model_builder import build_sam3_video_predictor

home = os.environ['HOME']
model_file_path = home + '/chLi/Model/SAM3/sam3/sam3.pt'
image_folder_path = home + "/chLi/Dataset/GS/haizei_1_v4/gs/images/"
mask_folder_path = home + "/chLi/Dataset/GS/haizei_1_v4/gs/masks/"

predictor = build_sam3_video_predictor(
    checkpoint_path=model_file_path,
    gpus_to_use=[0],
)

image_filename_list = os.listdir(image_folder_path)

valid_image_filename_list = []

for image_filename in image_filename_list:
    if image_filename.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
        continue

    valid_image_filename_list.append(image_filename)

valid_image_filename_list.sort()

valid_image_filename_list = valid_image_filename_list[:4]

print('start load images...')
image_list = []
for i in trange(len(valid_image_filename_list)):
    valid_image_file_path = image_folder_path + valid_image_filename_list[i]

    image = Image.open(valid_image_file_path)
    image_list.append(image)


def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=image_list,
    )
)
session_id = response["session_id"]

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
        frame_index=0,
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

os.makedirs(mask_folder_path, exist_ok=True)
print('start save mask...')
for i in trange(len(valid_image_filename_list)):
    mask = outputs_per_frame[i][0]
    cv2.imwrite(mask_folder_path + valid_image_filename_list[i], mask)

# finally, close the inference session to free its GPU resources
# (you may start a new session on another video)
_ = predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)

# after all inference is done, we can shutdown the predictor
# to free up the multi-GPU process group
predictor.shutdown()
