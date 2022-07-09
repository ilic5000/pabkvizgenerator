from time import sleep
import cv2
import numpy
import sys

fileName = 'potjera-e1320-frame.jpg'
#fileName = 'potera-srpska.png'

def nothing(x):
    # dummy
    pass

cv2.namedWindow("HSVTrackbars")

# green mask
cv2.createTrackbar("Lower-H", "HSVTrackbars", 31, 180, nothing)
cv2.createTrackbar("Lower-S", "HSVTrackbars", 23, 255, nothing)
cv2.createTrackbar("Lower-V", "HSVTrackbars", 0, 255, nothing)

cv2.createTrackbar("Upper-H", "HSVTrackbars", 84, 180, nothing)
cv2.createTrackbar("Upper-S", "HSVTrackbars", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "HSVTrackbars", 255, 255, nothing)


image = cv2.imread(fileName)

if image is None:
        print('Failed to load image file:', fileName)
        sys.exit(1)

hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

while True:
    l_h = cv2.getTrackbarPos("Lower-H", "HSVTrackbars")
    l_s = cv2.getTrackbarPos("Lower-S", "HSVTrackbars")
    l_v = cv2.getTrackbarPos("Lower-V", "HSVTrackbars")

    u_h = cv2.getTrackbarPos("Upper-H", "HSVTrackbars")
    u_s = cv2.getTrackbarPos("Upper-S", "HSVTrackbars")
    u_v = cv2.getTrackbarPos("Upper-V", "HSVTrackbars")

    lower_hsv = numpy.array([l_h, l_s, l_v])
    upper_hsv = numpy.array([u_h, u_s, u_v])

    # must be in the loop (reload empty image/frame)
    original_img_preview = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    mask = cv2.inRange(hsvImage, lower_hsv, upper_hsv)

    # Erode mask
    kernel = numpy.ones((5,5), numpy.uint8)
    mask = cv2.erode(mask, kernel)

    # if image is too big, scale it down for preview
    mask_half = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)
    kernel_half = numpy.ones((2,2), numpy.uint8)
    mask_half = cv2.erode(mask_half, kernel_half)

    #contours, _ = cv2.findContours(mask, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask_half, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)

    #for cnt in contours:
        #approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        #cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)

    for cnt in contours2:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4:
            print('dobro je')
        if area > 3500:
            cv2.drawContours(original_img_preview, [approx], 0, (0, 0, 255), 1)


    cv2.imshow('mask', mask_half)
    cv2.imshow('Original preview', original_img_preview)
    #cv2.imshow('Original full + contours', image)
    #cv2.imshow("image", mask)

    key = cv2.waitKey(1)
    if key == 27: # ESC
        break





#cv2.imwrite("mask-test.jpg", mask)     # save frame as JPEG file 
#cv2.imwrite("mask-original-test.jpg", image)

print('Done.')