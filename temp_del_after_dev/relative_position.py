import numpy as np
import cv2 as cv

img = cv.imread(r"temp_del_after_dev\fight.png", cv.IMREAD_COLOR)
print(img.shape)
cv.namedWindow("fight", cv.WINDOW_NORMAL)
xmin, ymin, w, h = cv.selectROI("fight", img)
print((xmin, ymin, w, h))
