import cv2
import numpy as np


def toMaskImage(mask: np.ndarray) -> np.ndarray:
    # 将bool类型的mask转为0/255的uint8, 再转换为RGB二值图
    mask_uint8 = (mask.astype("uint8") * 255)
    mask_image = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
    return mask_image


def toMaskedImage(
    image: np.ndarray,
    mask: np.ndarray,
    background_color: list = [255, 255, 255],
) -> np.ndarray:
    # 保证mask为2维，image为3通道
    if len(mask.shape) == 2:
        mask_3channel = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    else:
        mask_3channel = mask

    # 新建一个背景色填充的图像
    bg = np.full_like(image, background_color, dtype=image.dtype)
    # 用mask决定保留原图像素还是背景色
    masked_image = np.where(mask_3channel, image, bg)
    return masked_image
