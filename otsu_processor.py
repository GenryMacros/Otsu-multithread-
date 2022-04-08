import math

from cv2 import cv2
import numpy as np
from multiprocessing import Process, Array


class OtsuProcessor:

    def __init__(self, scale_factor_x, scale_factor_y, sigma, gauss_size, max_thresh, min_thresh):
        self.scale_factor_x = scale_factor_x
        self.scale_factor_y = scale_factor_y
        self.sigma = sigma
        self.gauss_size = gauss_size
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh

    def reset_scale_factors(self, scale_factor_x, scale_factor_y):
        self.scale_factor_x = scale_factor_x
        self.scale_factor_y = scale_factor_y

    def reset_gauss_blur_params(self, sigma, gauss_size):
        self.sigma = sigma
        self.gauss_size = gauss_size

    def reset_threshold_params(self, max_thresh, min_thresh):
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh

    def process_image(self, path, max_threads_count):
        image = np.array(cv2.imread(path))
        self.width, self.height, self.channels = image.shape
        image = self.grayscale(image)
        image = Array('i', image.reshape(image.shape[0] * image.shape[1] * 3), lock=False)

        image, hist = self.blur_hist_calculation(image, 40)
        threshold = self.calculate_otsu_threshold(10, hist)
        image = self.apply_threshold(image, threshold, 5)

        image = np.array(image[:]).reshape(self.width, self.height, 3)
        return image.astype(np.uint8)

    def grayscale(self, image):
        grayscaled = np.zeros(image.shape)
        x_size = image.shape[0]
        y_size = image.shape[1]

        for x in range(x_size):
            for y in range(y_size):
                gray = np.floor(image[x][y][0] * 0.299 + image[x][y][1] * 0.587 + image[x][y][2] * 0.114)
                grayscaled[x][y][0] = gray
                grayscaled[x][y][1] = gray
                grayscaled[x][y][2] = gray

        return grayscaled.astype(np.uint8)

    def blur_hist_calculation(self, image, threads_count):
        hist = Array('i', np.zeros(256, dtype=int), lock=False)
        blurred = Array('i', np.zeros(self.width * self.height * self.channels, dtype=int), lock=False)
        size = self.gauss_size // 2
        rows_per_thread = int(self.width / threads_count)
        x = 0
        threads = []
        while True:
            if x + rows_per_thread >= self.width:
                end = x + rows_per_thread
                thread = Process(target=self.blur_thread, args=(blurred, image, size, self.sigma, x, end - abs(end - self.width), self.width, self.height, hist))
                thread.start()
                threads.append(thread)
                break
            thread = Process(target=self.blur_thread, args=(blurred, image, size, self.sigma, x, x + rows_per_thread, self.width, self.height, hist))
            thread.start()
            threads.append(thread)
            x += rows_per_thread

        for thread in threads:
            thread.join()

        return blurred, hist

    def blur_thread(self, blurred, original, kernel, sigma, start, end, width, height, hist):

        for x in range(start, end):
            for y in range(height):
                for c in range(3):
                    total = 0
                    for i in range(x - kernel, min(x + kernel, width)):
                        if i < 0:
                            continue
                        for j in range(y - kernel, min(y + kernel, height)):
                            if j < 0:
                                continue
                            gauss_val = self.gauss_func(x - i, y - j, sigma)
                            total += original[(i * height + j) * 3 + c] * gauss_val
                    blurred[(x * height + y) * 3 + c] = int(total)

                index = (x * height + y) * 3
                hist[blurred[index]] += 1

    def gauss_func(self, x, y, sigma):
        e = np.e ** (-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
        main = 1 / (2 * np.pi * sigma ** 2)
        return main * e

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
