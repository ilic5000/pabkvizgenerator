from time import sleep
import cv2
import numpy
import sys
from skimage.metrics import structural_similarity

def does_template_exist(sourceImage, templateToFind, confidenceLevel):
    #img_gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)                                                                                                                
    res = cv2.matchTemplate(sourceImage, templateToFind, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= confidenceLevel:
        return True
    return False

fileName1 = 'examples/slagalica-test-diff-1.jpg'
fileName2 = 'examples/slagalica-test-diff-2.jpg'

image1 = cv2.imread(fileName1)
image2 = cv2.imread(fileName2)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)     
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)     


test = does_template_exist(image1, image2, 0.5)

diff = cv2.absdiff(image1, image2)
cv2.imshow('screen', diff)
image3 = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  

# cv2.waitKey()
# mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
# cv2.imshow('screen', mask)

# th = 1 
# imask =  mask>th

# canvas = numpy.zeros_like(image2, numpy.uint8)
# canvas[imask] = image2[imask]

# cv2.imshow('screen', canvas)

cv2.waitKey()