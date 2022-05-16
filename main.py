import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class SudokuSolver:
    def __init__(self):
        self.model = tf.keras.models.load_model("digit.model")

    def TractamentInicial(self, img):
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

    def DetectingGrid(self, img, contours):
        max_area = 0
        c = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 1000:
                if area > max_area:
                    max_area = area
                    best_cnt = i
                    img = cv2.drawContours(img, contours, c, (0, 255, 0), 3)
                c += 1
        mask = np.zeros((img.shape), np.uint8)
        cv2.drawContours(mask, [best_cnt], 0, 255, -1)
        plt.imshow(mask, cmap='gray')
        plt.show()
        return mask

    def solve(self,img):

        img = self.TractamentInicial(img)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = self.DetectingGrid(img, contours)

    def DividirTaulell(self, img):
        x, y = img.shape
        x_casella = int(x / 9)
        y_casella = int(y / 9)
        taulell = []
        for i in range(9):
            for j in range(9):
                posx = i * x_casella
                posy = j * y_casella
                taulell.append(img[posx:posx + x_casella, posy:posy + y_casella])
        return taulell


    def NumberRegonition(self, taulell):
        for i in taulell:
            img = cv2.resize(i, dsize=(28, 28))
            img = np.invert(np.array([img]))
            prediction = self.model.predict(img)
            if np.argmax(prediction) > 0:
                number  = np.argmax(prediction)
            else:
                number = None
            print(prediction)
            print("number: ", number)
            plt.imshow(img[0])
            plt.show()
