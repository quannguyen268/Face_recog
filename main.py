# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    bb = np.zeros((2, 2), dtype=np.int32)
    a = [[1, [1,2], 3], [2, [3,4], 5]]
    bb[0][:] = a[0][1][:]
    print(f'Hi, {bb}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

import cv2

img = cv2.imread('/home/quan/Pictures/2021/anh_the.png')
img = cv2.resize(img, (600,400), interpolation=cv2.INTER_AREA)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.imwrite('/home/quan/Pictures/2021/anh_the.png', img)