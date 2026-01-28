import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, List

from sam3.visualization_utils import prepare_masks_for_visualization
from sam3.model_builder import build_sam3_video_predictor

from sam_three.Method.utils import propagate_in_video

class Detector(object):
    def __init__(
        self,
        model_file_path: Optional[str]=None,
    ) -> None:
        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(
        self,
        model_file_path: str,
    ) -> bool:
        self.predictor = build_sam3_video_predictor(
            checkpoint_path=model_file_path,
            gpus_to_use=[0],
            max_num_objects=1,
        )
        return True

    def detectImages(
        self,
        image_list: List[Image],
    ) -> np.ndarray:
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=image_list,
            )
        )
        session_id = response["session_id"]

        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text="object",
            )
        )
        # out = response["outputs"]

        # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
        outputs_per_frame = propagate_in_video(self.predictor, session_id)

        # finally, we reformat the outputs for visualization and plot the outputs every 60 frames
        outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

        masks = []
        for i in range(len(outputs_per_frame)):
            mask = outputs_per_frame[i][0]  # HxW, bool np.ndarray
            masks.append(mask)
        # 合并为 KxHxW 的 np.ndarray
        masks = np.stack(masks, axis=0)

        # finally, close the inference session to free its GPU resources
        # (you may start a new session on another video)
        _ = self.predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        return masks

    def detectImageFiles(
        self,
        image_file_path_list: List[str],
    ) -> np.ndarray:
        image_list = []
        for image_file_path in tqdm(image_file_path_list):
            image = Image.open(image_file_path)
            image_list.append(image)

        return self.detectImages(image_list)

    def detectImageFolder(
        self,
        image_folder_path: str,
    ) -> np.ndarray:
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

        return self.detectImageFiles(valid_image_file_path_list)
