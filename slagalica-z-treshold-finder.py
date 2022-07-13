import cv2
import numpy
import sys

def nothing(x):
    # dummy
    pass

def preprocessBeforeOCR(imageToProcess, lower_bound, upper_bound, type, useBlurBefore, useBlurAfter):
    hsv = cv2.cvtColor(imageToProcess, cv2.COLOR_RGB2HSV)
    h, s, v1 = cv2.split(hsv)

    result = v1

    if useBlurBefore:
        result = cv2.GaussianBlur(v1,(5,5),0)

    # Can be played with... 
    result = cv2.threshold(v1, lower_bound, upper_bound, type)[1]
    
    if useBlurAfter:
        result = cv2.medianBlur(result, 3)

    return result
    
fileName1 = 'examples/Slagalica 01.01.2020. (1080p_25fps_H264-128kbit_AAC).mp4-21685-3.1-answer.jpg'
#fileName1 = 'examples/Slagalica 01.01.2020. (1080p_25fps_H264-128kbit_AAC).mp4-21685-2.1-question.jpg'

image = cv2.imread(fileName1)

#image2 = preprocessBeforeOCR(image, 147, 255, cv2.THRESH_BINARY)
#cv2.imshow('Image1', image2)

#cv2.waitKey()


cv2.namedWindow("tresholdTrackbars")
cv2.createTrackbar("lower_global_treshold", "tresholdTrackbars", 241, 255, nothing)
cv2.createTrackbar("upper_global_treshold", "tresholdTrackbars", 255, 255, nothing)
cv2.createTrackbar("gaussan_before_blur_on", "tresholdTrackbars", 0, 1, nothing)
cv2.createTrackbar("median_after_blur_on", "tresholdTrackbars", 0, 1, nothing)

while True:

    global_treshold_lower = cv2.getTrackbarPos("lower_global_treshold", "tresholdTrackbars")
    global_treshold_upper = cv2.getTrackbarPos("upper_global_treshold", "tresholdTrackbars")
    gaussan_blur_on = (cv2.getTrackbarPos("gaussan_before_blur_on", "tresholdTrackbars") == 1)
    median_blur_on = (cv2.getTrackbarPos("median_after_blur_on", "tresholdTrackbars") == 1)

    image1Processed = preprocessBeforeOCR(image.copy(), global_treshold_lower, global_treshold_upper, cv2.THRESH_BINARY, gaussan_blur_on, median_blur_on)
    cv2.imshow('Image1 global threshold', image1Processed)

    image1Processed2 = preprocessBeforeOCR(image.copy(), global_treshold_lower, global_treshold_upper, cv2.THRESH_BINARY + cv2.THRESH_OTSU, gaussan_blur_on, median_blur_on)
    cv2.imshow('Image1 global + otsu', image1Processed2)


    key = cv2.waitKey(1)

    if key == 27: # ESC
        break