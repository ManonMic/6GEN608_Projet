import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image_resized = imutils.resize(image, width=400)
cv2.imshow('image',image_resized)
cv2.waitKey(0)

images_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

cv2.destroyAllWindows()