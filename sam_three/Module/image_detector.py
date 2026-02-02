import os
import cv2
import numpy as np
from PIL import Image
from copy import deepcopy
from typing import Optional, Union

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_masks_to_frame, COLORS


class ImageDetector(object):
    def __init__(
        self,
        model_file_path: Optional[str]=None,
        device: str='cuda:0',
    ) -> None:
        if model_file_path is not None:
            self.loadModel(model_file_path, device)
        return

    def loadModel(
        self,
        model_file_path: str,
        device: str='cuda:0',
    ) -> bool:
        model = build_sam3_image_model(
            checkpoint_path=model_file_path,
            device=device,
        )
        self.processor = Sam3Processor(model, device=device, confidence_threshold=0.5)
        return True

    def detectImage(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
    ) -> dict:
        '''
        return:
            output["masks"], output["boxes"], output["scores"]
        '''
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inference_state = self.processor.set_image(image)

        self.processor.reset_all_prompts(inference_state)

        inference_state = self.processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )

        vis_image = deepcopy(image)

        # 使用 draw_masks_to_frame 在图像上绘制 mask 并写入 inference_state
        vis_frame = np.array(vis_image)[..., ::-1]
        masks = inference_state.get("masks", [])
        if len(masks) > 0:
            masks_np = np.stack([
                (m.squeeze(0).cpu().numpy() > 0).astype(np.uint8)
                for m in masks
            ])
            # 若 mask 尺寸与图像不一致则缩放到图像尺寸
            h, w = vis_frame.shape[:2]
            if masks_np.shape[1] != h or masks_np.shape[2] != w:
                masks_resized = np.stack([
                    cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    for m in masks_np
                ])
                masks_np = (masks_resized > 0).astype(np.uint8)
            n = len(masks_np)
            colors = (COLORS[:n] * 255).astype(np.uint8)
            vis_frame = draw_masks_to_frame(vis_frame, masks_np, colors)
        inference_state["image"] = vis_frame

        return inference_state

    def detectImageFile(
        self,
        image_file_path: str,
        prompt: str,
    ) -> Optional[dict]:
        if not os.path.exists(image_file_path):
            print('[ERROR][ImageDetector::detectImageFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return None

        image = Image.open(image_file_path)
        return self.detectImage(image, prompt)
