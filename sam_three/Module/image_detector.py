import os
import numpy as np
from PIL import Image
from copy import deepcopy
from typing import Optional, Union

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results


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
        self.processor = Sam3Processor(model, confidence_threshold=0.5)
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
        plot_results(vis_image, inference_state)
        inference_state['image'] = vis_image

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
