import numpy as np
import cv2


def show_correspondence(imgA, imgB, X1, Y1, X2, Y2, file_name='match_result'):
    Height = max(imgA.shape[0], imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]
    if len(imgA.shape) == 2:
        imgA = np.expand_dims(imgA, 2)
    if len(imgB.shape) == 2:
        imgB = np.expand_dims(imgB, 2)
    numColors = imgA.shape[2]
    newImg = np.zeros((Height, Width, numColors))
    newImg[:imgA.shape[0], :imgA.shape[1], :] = imgA
    newImg[:imgB.shape[0], imgA.shape[1]:, :] = imgB
    shiftX = imgA.shape[1]
    for i in range(X1.shape[0]):
        cur_color = np.random.rand(3)
        cv2.circle(newImg, (int(X1[i]), int(Y1[i])), 3, cur_color, 1)
        cv2.circle(newImg, (int(X2[i] + shiftX), int(Y2[i])), 3, cur_color, 1)
        cv2.line(newImg, (int(X1[i]), int(Y1[i])), (int(X2[i] + shiftX), int(Y2[i])), cur_color, 1)

    cv2.imwrite(file_name + '.jpg', newImg * 255.0)
