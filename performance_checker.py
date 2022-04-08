import time

import cv2
import numpy as np

from image_preprocessor import ImagePreprocessor
from otsu_processor import OtsuProcessor


class Timer:
    def __init__(self):
        self.locked_on = 0

    def start(self):
        self.locked_on = time.time()

    def get_elapsed(self):
        return time.time() - self.locked_on


class PerformanceChecker:

    def __init__(self, image_preprocessor, otsu_processor, image):
        self.iamge_processor = image_preprocessor
        self.otsu_processor = otsu_processor
        self.image = image
        self.timer = Timer()

    def get_preprocessor_single_process_time(self, image):
        self.timer.start()
        self.iamge_processor.prepare_image(image, 1)
        return self.timer.get_elapsed()

    def check_preprocessor_performance(self, path, processes):
        image = np.array(cv2.imread(path))
        single_thread_time = self.get_preprocessor_single_process_time(image)

        self.timer.start()
        self.iamge_processor.prepare_image(image, processes)
        multiprocess_time = self.timer.get_elapsed()

        return multiprocess_time / single_thread_time

    def get_otsu_single_process_time(self):
        test_hist = np.random.randint(256, size=256)
        self.timer.start()
        self.otsu_processor.process_image(1, with_preprocessing=False, test_hist=test_hist)
        return self.timer.get_elapsed()

    def check_otsu_performance(self, processes):
        self.otsu_processor.reset_image(self.image)
        single_thread_time = self.get_otsu_single_process_time()
        test_hist = np.random.randint(256, size=256)

        self.timer.start()
        self.otsu_processor.process_image(processes, with_preprocessing=False, test_hist=test_hist)
        multiprocess_time = self.timer.get_elapsed()

        return multiprocess_time / single_thread_time


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

    preprocessor_threads = 2
    otsu_threads = 2

    checker = PerformanceChecker(preprocessor, proc, image_path)
    otsu_performance = checker.check_otsu_performance(otsu_threads)
    preprocessor_performance = checker.check_preprocessor_performance(image_path, preprocessor_threads)

    print("Otsu performance with {} threads: {}".format(otsu_threads, otsu_performance))
    print("Preprocessor performance with {} threads: {}".format(preprocessor_threads, preprocessor_performance))





