
import cv2
import numpy as np

class GlobalThresholding:
    def __init__(self, image):
        # image must already be grayscale!
        self.image = image  

    def optimal_thresholding(self, initial_threshold=128, tolerance=0.5):
        gray_image = self.image
        threshold = initial_threshold if initial_threshold is not None else gray_image.mean()
        previous_threshold = 0

        while abs(threshold - previous_threshold) > tolerance:
            foreground = gray_image[gray_image >= threshold]
            background = gray_image[gray_image < threshold]

            mean_f = foreground.mean() if foreground.size > 0 else 0
            mean_b = background.mean() if background.size > 0 else 0

            previous_threshold = threshold
            threshold = (mean_f + mean_b) / 2

        binary_result = (gray_image >= threshold).astype(np.uint8) * 255
        return binary_result

    def otsu_thresholding(self):
        gray_image = self.image
        histogram, _ = np.histogram(gray_image.ravel(), bins=256, range=(0, 256))
        total_pixels = gray_image.size
        sum_all = np.dot(np.arange(256), histogram)

        max_variance = 0
        threshold = 0
        background_w = 0
        background_sum = 0

        for t in range(256):
            background_w += histogram[t]
            if background_w == 0:
                continue

            foreground_w = total_pixels - background_w
            if foreground_w == 0:
                break

            background_sum += t * histogram[t]

            mean_b = background_sum / background_w
            mean_f = (sum_all - background_sum) / foreground_w

            variance_between = (
                background_w * foreground_w * (mean_b - mean_f) ** 2
            )

            if variance_between > max_variance:
                max_variance = variance_between
                threshold = t

        binary_result = (gray_image >= threshold).astype(np.uint8) * 255
        return binary_result
