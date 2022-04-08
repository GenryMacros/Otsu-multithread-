import math

from cv2 import cv2
import numpy as np
from multiprocessing import Process, Array


class OtsuProcessor:

    def __init__(self, image_path, proprocessor, max_thresh, min_thresh):
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.path = image_path
        self.proprocessor = proprocessor

    def reset_threshold_params(self, max_thresh, min_thresh):
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh

    def reset_image(self, image_path):
        self.path = image_path

    def process_image(self, threads_count, with_preprocessing=True, test_hist=None):
        image = np.array(cv2.imread(self.path))
        self.width, self.height, self.channels = image.shape
        if with_preprocessing:
            image, hist = self.proprocessor.prepare_image(image, threads_count)
        else:
            hist = test_hist
            image = Array('i', image.reshape(image.shape[0] * image.shape[1] * 3), lock=False)
        threshold = self.calculate_otsu_threshold(threads_count, hist)
        image = self.apply_threshold(image, threshold, threads_count)

        image = np.array(image[:]).reshape(self.width, self.height, 3)
        return image.astype(np.uint8)

    def otsu_thread(self, hist, start_i, end_i, within, image_pixels):
        for i in range(start_i, end_i):
            x, y = np.split(hist, [i])
            w0 = np.sum(x) / image_pixels
            w1 = 1 - w0

            u0 = np.sum([j * (t / image_pixels) for j, t in enumerate(x)]) / w0
            u1 = np.sum([j * (t / image_pixels) for j, t in enumerate(y)]) / w1

            q1 = np.sum([((j - u0) ** 2) * (t / image_pixels) for j, t in enumerate(x)]) / w0
            q2 = np.sum([((j - u1) ** 2) * (t / image_pixels) for j, t in enumerate(y)]) / w1
            res = w0 * q1 + w1 * q2
            if math.isnan(res):
                res = np.inf
            within[i] = res

    def calculate_otsu_threshold(self, thread_count, hist):
        nums_per_thread = int(256 / thread_count)
        within = Array('f', np.zeros(256, dtype=float), lock=False)
        x = 0
        processes = []
        while True:
            if x + nums_per_thread >= 256:
                end = x + nums_per_thread
                process = Process(target=self.otsu_thread, args=(hist, x, end - abs(end - 256), within, self.width * self.height))
                process.start()
                processes.append(process)
                break
            end = x + nums_per_thread
            process = Process(target=self.otsu_thread, args=(hist, x, end, within, self.width * self.height))
            process.start()
            processes.append(process)
            x += nums_per_thread

        for process in processes:
            process.join()

        within = np.array(within[:])
        return np.argmin(within)

    def threshold_thread(self, image, start_x, end_x, height, threshold, maxcol, mincol):
        for x in range(start_x, end_x):
            for y in range(height):
                index = (x * height + y) * 3

                if image[index] > threshold[0]:
                    image[index + 0] = maxcol[0]
                    image[index + 1] = maxcol[1]
                    image[index + 2] = maxcol[2]
                else:
                    image[index + 0] = mincol[0]
                    image[index + 1] = mincol[1]
                    image[index + 2] = mincol[2]

    def apply_threshold(self, image, threshold, threads_count):
        maxcol = Array('i', self.max_thresh, lock=False)
        mincol = Array('i', self.min_thresh, lock=False)
        threshold = Array('i', [threshold], lock=False)
        rows_per_thread = int(self.width / threads_count)
        x = 0
        processes = []
        while True:
            if x + rows_per_thread >= self.width:
                end = x + rows_per_thread
                process = Process(target=self.threshold_thread,
                                  args=(image, x, end - abs(end - self.width), self.height, threshold, maxcol, mincol))
                process.start()
                processes.append(process)
                break
            process = Process(target=self.threshold_thread,
                              args=(image, x, x + rows_per_thread, self.height, threshold, maxcol, mincol))
            process.start()
            processes.append(process)
            x += rows_per_thread

        for process in processes:
            process.join()

        return image
