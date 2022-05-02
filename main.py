import cv2
import matplotlib.pyplot as plt
import numpy as np


def TractamentInicial(img):
    # APLIQUEM ADAPTIVE THRESHOLD
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 199, 5)
    plt.imshow(thresh, cmap="gray")
    plt.show()

    # DILATACIÓ

    kernel = np.ones((2, 2), 'uint8')
    dilate_img = cv2.dilate(thresh, kernel, iterations=1)
    plt.imshow(dilate_img, cmap="gray")
    plt.show()

    return dilate_img


def DetectingGrid(img):
    max = -1
    maxPt = (0, 0)

    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    for y in range(0, h):
        for x in range(0, w):

            if img[y, x] >= 128:
                area = cv2.floodFill(img, mask, (x, y), (0, 0, 64))
    return area


def main():
    img = cv2.imread("images/sudoku1.jfif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap="gray")
    plt.show()

    img = TractamentInicial(img)
    grid = DetectingGrid(img)
    plt.imshow(grid, cmap="gray")
    plt.show()


main()
