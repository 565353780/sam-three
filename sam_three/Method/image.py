import cv2
import numpy as np


def toMaskImage(mask: np.ndarray) -> np.ndarray:
    # 将bool类型的mask转为0/255的uint8, 再转换为RGB二值图
    mask_uint8 = (mask.astype("uint8") * 255)
    mask_image = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
    return mask_image


def toMaskedImage(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # 将mask扩展为3通道（RGB）
    mask_3channel = mask[:, :, np.newaxis] if len(mask.shape) == 2 else mask

    # 使用mask抠图：将mask为False的区域设为0（黑色背景）
    masked_image = image * mask_3channel
    return masked_image


