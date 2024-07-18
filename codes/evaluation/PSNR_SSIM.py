import cv2
from utils import util

original_image = cv2.imread('../datasets/0801/0801.png')
processed_image = cv2.imread('../results/0801/0801-DNA-QLC.png')


def cal_pnsr_ssim(original_image, processed_image):
    # calculate PSNR and SSIM
    original_image = original_image / 255.
    processed_image = processed_image / 255.

    if original_image is None or processed_image is None:
        print("One or both images could not be loaded.")
    else:
        # 将两个图像调整到相同的尺寸
        height = min(original_image.shape[0], processed_image.shape[0])
        width = min(original_image.shape[1], processed_image.shape[1])

        original_image_resized = cv2.resize(original_image, (width, height))
        processed_image_resized = cv2.resize(processed_image, (width, height))

    psnr = util.calculate_psnr(original_image_resized * 255, processed_image_resized * 255)
    ssim = util.calculate_ssim(original_image_resized * 255, processed_image_resized * 255)
    return ssim, psnr


ssim, psnr = cal_pnsr_ssim(original_image, processed_image)
print("psnr:%.3f" % (psnr))
print("ssim:%.3f" % (ssim))
