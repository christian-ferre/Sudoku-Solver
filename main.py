import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class SudokuSolver:
    def __init__(self):
        self.model = tf.keras.models.load_model("digit.model")

    def solve(self,img):
        """
        img = self.TractamentInicial(img)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = self.DetectingGrid(img, contours)
       """

        taulell = self.DividirTaulell(img)
        sudoku = self.NumberRegonition(taulell)
        print(sudoku)

    def DividirTaulell(self,img):
        x, y = img.shape
        x_casella = int(x / 9)
        y_casella = int(y / 9)
        taulell = []
        for i in range(9):
            fila = []
            for j in range(9):
                posx = i * x_casella
                posy = j * y_casella
                fila.append(img[posx:posx + x_casella, posy:posy + y_casella])
            taulell.append(fila)
        return taulell

    def NumberRegonition(self, taulell):
        sudoku = []
        for i in range(len(taulell)):
            fila = []
            for j in range(len(taulell[i])):
                img = cv2.resize(taulell[i][j], dsize=(28, 28))
                img = np.invert(np.array([img]))
                prediction = self.model.predict(img)
                fila.append(np.argmax(prediction))
            sudoku.append(fila)
        return sudoku

    def solveSudoku(self):
        pass


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

def getContours(canny, img):
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20000:
            cv2.drawContours(img, cnt, -1, (0,255,0), 2)

            p = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * p, True)
            ax = approx.item(0)
            ay = approx.item(1)
            bx = approx.item(2)
            by = approx.item(3)
            cx = approx.item(4)
            cy = approx.item(5)
            dx = approx.item(6)
            dy = approx.item(7)

            w, h = 900, 900

            pts1 = np.float32([[bx, by],[ax, ay], [cx, cy], [dx,dy]])
            pts2 = np.float32([[0, 0],[w, 0], [0, h], [w,h]])

            m = cv2.getPerspectiveTransform(pts1, pts2)
            pres = cv2.warpPerspective(img, m, (w, h))
            conto = cv2.cvtColor(pres, cv2.COLOR_BGR2GRAY)
            
            cv2.imshow('Conto', conto)
            a = SudokuSolver()
            a.solve(conto)
            return conto

def TractamentInicial(img):
    # APLIQUEM ADAPTIVE THRESHOLD
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 5)

    # DILATACIÓ
    kernel = np.ones((1, 1), 'uint8')
    dilate_img = cv2.dilate(thresh, kernel, iterations=1)

    return dilate_img


def DetectGrid(img, contours):
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
    return mask, best_cnt


def DetectCorners(img, cnt):  
    p = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * p, True)
    ax = approx.item(0)
    ay = approx.item(1)
    bx = approx.item(2)
    by = approx.item(3)
    cx = approx.item(4)
    cy = approx.item(5)
    dx = approx.item(6)
    dy = approx.item(7)

    w, h = 900, 900

    pts1 = np.float32([[ax, ay],[bx, by], [cx, cy], [dx,dy]])
    pts2 = np.float32([[w, 0],[0, 0], [0, h], [w,h]])

    m = cv2.getPerspectiveTransform(pts1, pts2)
    pres = cv2.warpPerspective(img, m, (w, h))
    conto = cv2.cvtColor(pres, cv2.COLOR_BGR2GRAY)
    
    plt.imshow(conto, cmap='gray')
    plt.show()
    return conto


def main():
    original = cv2.imread("images/sudoku1.jfif")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap="gray")
    plt.show()

    img = TractamentInicial(gray)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask, best_cnt = DetectGrid(original, contours)
    
    res = DetectCorners(original, best_cnt)
    a = SudokuSolver()
    a.solve(res)

    




main()

