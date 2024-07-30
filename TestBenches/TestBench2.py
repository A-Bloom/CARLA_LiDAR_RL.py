# This TestBench is a simplified version of how create_lidar_plane() takes an array of points and
# turns it into a point grid or image.

import numpy as np
import cv2

size = 9
multiplier = 20

x = np.array([-2, -2, -2, -3, -3, -3, -3, -3, -2, -2, -1, 0, 0, 1, 2, 3, 4, 4, 4, 4, 3, 3, 3, 3], dtype=int)
y = np.array([-4, -3, -2, -2, -1, 0, 1, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 1, 0, -1, -1, -2, -3, -4], dtype=int)

a = np.zeros((size*multiplier, size*multiplier))

x = ((((size-1)/2)+x)*multiplier).astype(dtype=int)
y = ((((size-1)/2)-y)*multiplier).astype(dtype=int)

a[y, x] = 1

print(a)

blanks = np.zeros((size*multiplier, size*multiplier))

pic = np.dstack((blanks, blanks, a * 255))

cv2.namedWindow('View', cv2.WINDOW_AUTOSIZE)
cv2.imshow('View', pic)
cv2.waitKey(1)

while True:
    cv2.imshow('View', pic)

    if cv2.waitKey(1) == ord('q'):
        break
