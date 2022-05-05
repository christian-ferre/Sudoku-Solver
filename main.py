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

    kernel = np.ones((1, 1), 'uint8')
    dilate_img = cv2.dilate(thresh, kernel, iterations=1)
    plt.imshow(dilate_img, cmap="gray")
    plt.show()

    return dilate_img


def DetectingGrid(img, contours):
    max_area = 0
    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = i
                img = cv2.drawContours(img, contours, c, (0, 255, 0), 3)
            c+=1
    mask = np.zeros((img.shape),np.uint8)
    cv2.drawContours(mask,[best_cnt],0,255,-1)
    plt.imshow(mask, cmap='gray')
    plt.show()
    return mask


def main():
    img = cv2.imread("images/sudoku1.jfif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap="gray")
    plt.show()

    img = TractamentInicial(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = DetectingGrid(img, contours)



main()
