from time import sleep
import cv2
import numpy
import sys

fileName = 'potjera-e1320-frame.jpg'
#fileName = 'potera-srpska.png'
#fileName = 'prosta-slika-test.png'

def nothing(x):
    # dummy
    pass

cv2.namedWindow("HSVTrackbars")

# green mask (hsv 31,23,0 to 84, 255, 255)
cv2.createTrackbar("Lower-H", "HSVTrackbars", 31, 180, nothing)
cv2.createTrackbar("Lower-S", "HSVTrackbars", 23, 255, nothing)
cv2.createTrackbar("Lower-V", "HSVTrackbars", 0, 255, nothing)

cv2.createTrackbar("Upper-H", "HSVTrackbars", 84, 180, nothing)
cv2.createTrackbar("Upper-S", "HSVTrackbars", 255, 255, nothing)
cv2.createTrackbar("Upper-V", "HSVTrackbars", 255, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

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

    original_img_previewHeight, original_img_previewWidth, channels = original_img_preview.shape 
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
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        #print('len contours2 %d' %len(contours2)) 
        if len(approx) == 4:
            #print('len approx %d' %len(approx))
            if area > 5000:
                #print('area %d' %area)
                cv2.drawContours(original_img_preview, [approx], 0, (0, 0, 255), 2)

                #https://www.geeksforgeeks.org/find-co-ordinates-of-contours-using-opencv-python/

                n = approx.ravel() 
                i = 0
                ymin = original_img_previewHeight
                ymax = 0
                xmin = original_img_previewWidth
                xmax = 0

                for j in n :
                    if(i % 2 == 0):
                        x = n[i]
                        y = n[i + 1]

                        if y < ymin:
                            ymin = y

                        if y > ymax:
                            ymax = y

                        if x < xmin:
                            xmin = x

                        if x > xmax:
                            xmax = x
                        
                        textCoordinate = str(x) + " " + str(y) 
                        cv2.putText(original_img_preview, textCoordinate, (x, y), 
                                    font, 0.5, (0, 0, 255)) 
                    i = i + 1
                
                textYMinMax = "ymin: " + str(ymin) + " ymax: " + str(ymax)
                textXMinMax = "xmin: " + str(xmin) + " xmax: " + str(xmax)
                cv2.putText(original_img_preview, textYMinMax, (0, 100), 
                            font, 0.5, (0, 0, 255)) 
                cv2.putText(original_img_preview, textXMinMax, (0, 120), 
                            font, 0.5, (0, 0, 255)) 


    lowerThirdYUpper = 2 * int(original_img_previewHeight/3)
    line_thickness = 2
    cv2.line(original_img_preview, (0, lowerThirdYUpper), (original_img_previewWidth, lowerThirdYUpper), (0, 255, 0), thickness=line_thickness)
    
    lowerEightXLower = int(original_img_previewWidth/8.5)
    line_thickness = 2
    cv2.line(original_img_preview, (lowerEightXLower, lowerThirdYUpper), (lowerEightXLower, original_img_previewHeight), (0, 255, 0), thickness=line_thickness)

    lowerEightXUpper = int(7.5 * int(original_img_previewWidth/8.5))
    line_thickness = 2
    cv2.line(original_img_preview, (lowerEightXUpper, lowerThirdYUpper), (lowerEightXUpper, original_img_previewHeight), (0, 255, 0), thickness=line_thickness)

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