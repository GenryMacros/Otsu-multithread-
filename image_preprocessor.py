import math

import numpy as np
from multiprocessing import Process, Array


class ImagePreprocessor:

    def __init__(self, scale_factor_x, scale_factor_y, sigma, gauss_size):
        self.scale_factor_x = scale_factor_x
        self.scale_factor_y = scale_factor_y
        self.sigma = sigma
        self.gauss_size = gauss_size

    def prepare_image(self, image, thread_count):
        self.width, self.height, self.channels = image.shape
        image = self.grayscale(image)
        image = Array('i', image.reshape(image.shape[0] * image.shape[1] * 3), lock=False)

        image, hist = self.blur_hist_calculation(image, thread_count)
        return image, hist

    def reset_scale_factors(self, scale_factor_x, scale_factor_y):
        self.scale_factor_x = scale_factor_x
        self.scale_factor_y = scale_factor_y

    def reset_gauss_blur_params(self, sigma, gauss_size):
        self.sigma = sigma
        self.gauss_size = gauss_size

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

    def scale(original_img, new_h, new_w):

        old_h, old_w, c = original_img.shape

        resized = np.zeros((new_h, new_w, c))

        w_scale_factor = old_w / new_w if new_h != 0 else 0
        h_scale_factor = old_h / new_h if new_w != 0 else 0
        for i in range(new_h):
            for j in range(new_w):

                x = i * h_scale_factor
                y = j * w_scale_factor

                x_floor = math.floor(x)
                x_ceil = min(old_h - 1, math.ceil(x))
                y_floor = math.floor(y)
                y_ceil = min(old_w - 1, math.ceil(y))

                if (x_ceil == x_floor) and (y_ceil == y_floor):
                    q = original_img[int(x), int(y), :]
                elif x_ceil == x_floor:
                    q1 = original_img[int(x), int(y_floor), :]
                    q2 = original_img[int(x), int(y_ceil), :]
                    q = q1 * (y_ceil - y) + q2 * (y - y_floor)
                elif y_ceil == y_floor:
                    q1 = original_img[int(x_floor), int(y), :]
                    q2 = original_img[int(x_ceil), int(y), :]
                    q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))
                else:
                    v1 = original_img[x_floor, y_floor, :]
                    v2 = original_img[x_ceil, y_floor, :]
                    v3 = original_img[x_floor, y_ceil, :]
                    v4 = original_img[x_ceil, y_ceil, :]

                    q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                    q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                    q = q1 * (y_ceil - y) + q2 * (y - y_floor)

                resized[i, j, :] = q
        return resized.astype(np.uint8)
