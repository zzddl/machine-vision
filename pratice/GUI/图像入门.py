import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../temp/lenna.png', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('../temp/messigray.png', img)
