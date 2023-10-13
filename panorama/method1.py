import glob
import os
import cv2
img_list = glob.glob('./imgs2/*.jpg')

imgs = [cv2.imread(path) for path in img_list]
stitcher = cv2.Stitcher_create()
result = stitcher.stitch(imgs)[1]
cv2.imwrite("Panorama.jpg",result)
