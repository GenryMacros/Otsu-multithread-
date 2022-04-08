import numpy as np
from cv2 import cv2
import math
import time
from otsu_processor import OtsuProcessor

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


if __name__ == '__main__':
    image_path = "C:/Users/Light/Pictures/test6.jpg"
    scale_factor_x = 400
    scale_factor_y = 400
    sigma = 1.2
    gauss_blur_size = 5
    max_thresh = [255, 255, 255]
    min_thresh = [0, 0, 0]
    proc = OtsuProcessor(scale_factor_x, scale_factor_y, sigma, gauss_blur_size, max_thresh, min_thresh)
    image = proc.process_image(image_path, 40)
    cv2.imshow("Test1", image)
    cv2.waitKey(0)

