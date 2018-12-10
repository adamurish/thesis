import cv2
import numpy as np;

# # face_cas = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# stop_cas = cv2.CascadeClassifier('stop_cascade/cascade.xml')
#
# img = cv2.imread('stop_test.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # faces = face_cas.detectMultiScale(gray, 1.3, 5)
# stops = stop_cas.detectMultiScale(gray, 1.3, 5)
# print(stops)
#
# for (x, y, w, h) in stops:
#     img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#
# # gray = np.float32(gray)
# # dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# #
# # dst = cv2.dilate(dst, None)
# # img[dst > 0.1*dst.max()] = [0, 0, 255]
# # print(img.shape)
#
# # corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
# # corners = np.int0(corners)
# #
# # for i in corners:
# #     x, y = i.ravel()
# #     cv2.circle(img, (x, y), 3, 255, -1)
#
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
try_a = [1,2,3,4]
print(try_a[-1])
