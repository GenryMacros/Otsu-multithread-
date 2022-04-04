import numpy as np
from cv2 import cv2
import math


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


def grayscale_image(image):
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


def calculate_otsu_threshold(image):
    hist = np.array(calculate_hist(image, 0))
    within = []
    for i in range(len(hist)):
        x, y = np.split(hist, [i])
        w0 = np.sum(x) / (image.shape[0]*image.shape[1])
        w1 = 1 - w0

        u0 = np.sum([j*(t / (image.shape[0]*image.shape[1])) for j, t in enumerate(x)]) / w0
        u1 = np.sum([j*(t / (image.shape[0]*image.shape[1])) for j, t in enumerate(y)]) / w1

        q1 = np.sum([((j-u0)**2) * (t / (image.shape[0]*image.shape[1])) for j, t in enumerate(x)]) / w0
        q2 = np.sum([((j-u1)**2) * (t / (image.shape[0]*image.shape[1])) for j, t in enumerate(y)]) / w1
        res = w0*q1 + w1*q2
        if math.isnan(res):
            res = np.inf
        within.append(res)

    m = np.argmin(within)

    return m


def apply_threshold(image, threshold, maxval, minval):
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j][0] > threshold:
                image[i][j][0] = maxval[0]
                image[i][j][1] = maxval[1]
                image[i][j][2] = maxval[2]
            else:
                image[i][j][0] = minval[0]
                image[i][j][1] = minval[1]
                image[i][j][2] = minval[2]
    return image


def calculate_hist(image, channel):
    hist = np.zeros(256, dtype=int)

    for i in range(len(image)):
        for j in range(len(image[i])):
            hist[image[i][j][channel]] += 1

    return hist


def blur(image, sigma, size):
    width, height, channels = image.shape
    blurred = np.zeros(image.shape)
    size = size // 2
    for x in range(width):
        for y in range(height):
            for c in range(channels):
                total = 0
                for i in range(x - size, min(x + size, width)):
                    if i < 0:
                        continue
                    for j in range(y - size, min(y + size, height)):
                        if j < 0:
                            continue
                        gauss_val = gauss_func(x - i, y - j, sigma)
                        total += image[i][j][c] * gauss_val
                blurred[x][y][c] = total

    return blurred.astype(np.uint8)


def gauss_func(x, y, sigma):
    e = np.e**(-((x**2 + y**2) / (2 * sigma**2)))
    main = 1 / (2 * np.pi * sigma**2)
    return main * e


if __name__ == '__main__':
    image_path = "C:/Users/Light/Pictures/kind0.png"
    image = cv2.imread(image_path)
    image = np.array(image)
    scale_factor_x = 400
    scale_factor_y = 400
    sigma = 1.2
    gauss_blur_size = 5
    max_thresh = [255, 255, 255]
    min_thresh = [0, 0, 0]
    #image = scale(image, scale_factor_x, scale_factor_y)
    #image = blur(image, sigma, gauss_blur_size)
    image = grayscale_image(image)
    threshold = calculate_otsu_threshold(image)
    image = apply_threshold(image, threshold, max_thresh, min_thresh)
    cv2.imshow("Test1", image)
    cv2.waitKey(0)

