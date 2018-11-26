import cv2
from matplotlib import pyplot as plt

print(cv2)
img = cv2.imread('test.jpg')

cv2.imshow('test', cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REPLICATE))
cv2.waitKey(0)
cv2.destroyAllWindows()
