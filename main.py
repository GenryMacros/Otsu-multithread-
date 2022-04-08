from cv2 import cv2
from otsu_processor import OtsuProcessor
from image_preprocessor import ImagePreprocessor

if __name__ == '__main__':
    image_path = "/home/light_frosted/Pictures/smoothed.png"
    scale_factor_x = 400
    scale_factor_y = 400
    sigma = 1.2
    gauss_blur_size = 5
    max_thresh = [255, 255, 255]
    min_thresh = [0, 0, 0]

    preprocessor = ImagePreprocessor(scale_factor_x, scale_factor_y, sigma, gauss_blur_size)
    proc = OtsuProcessor(image_path, preprocessor, max_thresh, min_thresh)
    image = proc.process_image(40)
    cv2.imshow("Test1", image)
    cv2.waitKey(0)

